import pytest
from sqlalchemy import event

from models.User import User


@pytest.fixture(autouse=True)
def activate_test_users():
    """Keep authentication fixtures active unless a test explicitly disables them."""
    def set_active(user, args, kwargs):
        if "is_active" not in kwargs:
            user.is_active = True

    event.listen(User, "init", set_active)
    yield
    event.remove(User, "init", set_active)
