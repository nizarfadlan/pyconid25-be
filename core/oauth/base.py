from datetime import datetime, timedelta
import secrets
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TypedDict, Union
from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse
import jwt
import pytz
from sqlalchemy import func, or_, select
from models.User import User
from settings import ALGORITHM, FRONTEND_BASE_URL, OAUTH_AUTO_LINK_BY_VERIFIED_EMAIL, SECRET_KEY, TZ
from authlib.integrations.starlette_client import OAuth
from sqlalchemy.orm import Session


class UserInfoResponse(TypedDict):
    id: str
    username: str
    email: Optional[str]
    name: Optional[str]


class HandleOauthResponse(TypedDict):
    user: User
    is_new_user: bool
    provider_user_info: UserInfoResponse


class OAuthTokenResponse(TypedDict, total=False):
    access_token: str
    refresh_token: Optional[str]
    expires_in: Optional[int]
    token_type: Optional[str]
    scope: Optional[str]


class BaseOAuthService(ABC):
    """Base OAuth service that must be inherited by all OAuth providers."""

    def __init__(self):
        self.oauth = OAuth()
        self._register_provider()

    @abstractmethod
    def _register_provider(self):
        """Register OAuth providers using authlib."""
        pass

    @abstractmethod
    async def _get_user_info(
        self, client, token: OAuthTokenResponse
    ) -> UserInfoResponse:
        """Get user info from the API provider."""
        pass

    @abstractmethod
    def _get_provider_name(self) -> str:
        """Return provider name (github, google, discord, etc)"""
        pass

    @abstractmethod
    def _get_user_by_provider_id(self, db: Session, provider_id: str) -> Optional[User]:
        """Find user by provider-specific ID field"""
        pass

    @abstractmethod
    def _update_user_provider_info(
        self, user: User, user_info: UserInfoResponse
    ) -> User:
        """Update user dengan provider-specific fields"""
        pass

    @abstractmethod
    def _set_user_provider_fields(
        self, user_data: dict, user_info: UserInfoResponse, provider_id: str
    ) -> dict:
        """Set provider-specific fields untuk user baru"""
        pass

    def _create_oauth_state(
        self, redirect_uri: Optional[str] = None, provider: Optional[str] = None
    ) -> str:
        payload = {
            "redirect_uri": redirect_uri,
            "provider": provider,
            "nonce": secrets.token_urlsafe(16),
            "exp": datetime.now(tz=pytz.timezone("UTC")) + timedelta(minutes=10),
            "iat": datetime.now(tz=pytz.timezone("UTC")),
        }
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    def _verify_oauth_state(self, state: str) -> dict:
        try:
            payload = jwt.decode(state, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=400, detail="OAuth state expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=400, detail="Invalid OAuth state")

    def _get_redirect_uri(self, provider_name: str) -> str:
        if FRONTEND_BASE_URL is None:
            raise HTTPException(status_code=500, detail="FRONTEND_BASE_URL is not set")
        return f"{FRONTEND_BASE_URL.rstrip('/')}/auth/{provider_name}/callback/"

    @staticmethod
    def _normalize_email(email: Optional[str]) -> Optional[str]:
        if not email:
            return None
        normalized_email = email.strip().lower()
        return normalized_email or None

    def _get_user_by_email_identity(self, db: Session, email: str) -> Optional[User]:
        stmt = select(User).where(
            or_(
                func.lower(User.email) == email,
                func.lower(User.google_email) == email,
                func.lower(User.username) == email,
            )
        )
        return db.execute(stmt).scalar()

    async def initiate_oauth(
        self, request: Request, follow_redirect: Optional[bool] = False
    ) -> Union[RedirectResponse, str]:
        """
        Initiate OAuth flow.

        The same general logic for all providers.
        """
        provider_name = self._get_provider_name()

        if not hasattr(self.oauth, provider_name):
            raise HTTPException(
                status_code=500,
                detail=f"OAuth provider '{provider_name}' is not properly configured",
            )

        client = getattr(self.oauth, provider_name)

        if FRONTEND_BASE_URL is None:
            raise HTTPException(status_code=500, detail="FRONTEND_BASE_URL is not set")

        try:
            redirect_uri = self._get_redirect_uri(provider_name)

            state = self._create_oauth_state(
                redirect_uri=redirect_uri, provider=provider_name
            )

            if follow_redirect:
                return await client.authorize_redirect(
                    request, redirect_uri, state=state
                )

            authorization_url = await client.create_authorization_url(
                redirect_uri, state=state
            )
            return authorization_url.get("url", None)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initiate {provider_name} OAuth: {str(e)}",
            )

    async def handle_verified(
        self, request: Request, db: Session, code: str, state: str
    ) -> HandleOauthResponse:
        """
        Handle OAuth callbacks and create/update users.

        The same general logic for all providers.
        Provider-specific logic is delegated to abstract methods.
        """
        provider_name = self._get_provider_name()

        if not hasattr(self.oauth, provider_name):
            raise HTTPException(
                status_code=500,
                detail=f"OAuth provider '{provider_name}' is not properly configured",
            )

        client = getattr(self.oauth, provider_name)

        if not code or not state:
            raise HTTPException(
                status_code=400,
                detail="Missing 'code' or 'state' in OAuth callback request",
            )

        try:
            state_payload = self._verify_oauth_state(state)
            redirect_uri = state_payload.get("redirect_uri")
            expected_redirect_uri = self._get_redirect_uri(provider_name)
            if state_payload.get("provider") != provider_name:
                raise HTTPException(status_code=400, detail="Invalid OAuth state provider")
            if redirect_uri != expected_redirect_uri:
                raise HTTPException(status_code=400, detail="Invalid OAuth redirect URI")
        except HTTPException as e:
            raise e

        try:
            # Get access token
            token = await client.fetch_access_token(
                code=code, redirect_uri=redirect_uri
            )

            # Get user info from provider
            user_info = await self._get_user_info(client, token)

            # Find or create user
            user, is_new_user = await self._find_or_create_user(db, user_info)

            return {
                "user": user,
                "is_new_user": is_new_user,
                "provider_user_info": user_info,
            }

        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            traceback.print_exc()
            raise HTTPException(
                status_code=400,
                detail=f"OAuth verification for {provider_name} failed: {str(e)}",
            )

    async def _find_or_create_user(
        self, db: Session, user_info: UserInfoResponse
    ) -> Tuple[User, bool]:
        """
        Common user finding/creation logic.
        """
        provider_name = self._get_provider_name()
        provider_id = user_info.get("id")
        provider_username = user_info.get("username")
        provider_email = self._normalize_email(user_info.get("email"))
        user_info = {**user_info, "email": provider_email}

        if not provider_id:
            raise HTTPException(
                status_code=400, detail=f"Missing {provider_name} user identifier"
            )

        existing_user = self._get_user_by_provider_id(db, provider_id)

        if existing_user:
            updated_user = self._update_user_provider_info(existing_user, user_info)
            updated_user.updated_at = datetime.now(pytz.timezone(TZ))
            db.commit()
            return updated_user, False

        user: Optional[User] = None

        if OAUTH_AUTO_LINK_BY_VERIFIED_EMAIL and provider_email:
            user = self._get_user_by_email_identity(db, provider_email)

        is_new_user = False

        if not user:
            if not provider_email:
                raise HTTPException(
                    status_code=400,
                    detail=f"No verified {provider_name.title()} email available",
                )

            if not OAUTH_AUTO_LINK_BY_VERIFIED_EMAIL:
                existing = self._get_user_by_email_identity(db, provider_email)
                if existing:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Email already registered. Please sign in and link your {provider_name.title()} account from settings.",
                    )

            user_data = {
                "username": provider_email if provider_email else provider_username,
                "password": None,
                "is_active": True,
                "email": provider_email,
                "created_at": datetime.now(pytz.timezone(TZ)),
                "updated_at": datetime.now(pytz.timezone(TZ)),
            }

            user_data = self._set_user_provider_fields(
                user_data, user_info, provider_id
            )

            user = User(**user_data)
            db.add(user)
            db.flush()
            is_new_user = True

        else:
            user = self._update_user_provider_info(user, user_info)
            user.updated_at = datetime.now(pytz.timezone(TZ))
            db.add(user)

        db.commit()
        return user, is_new_user
