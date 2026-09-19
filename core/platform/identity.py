"""Owner context for trusted in-process calls; HTTP always requires an explicit user."""
from contextlib import contextmanager
from contextvars import ContextVar
import re

LEGACY_USER = 'legacy-researcher-owner'
_user = ContextVar('researcher_user', default=LEGACY_USER)


def current_user():
    return _user.get()


@contextmanager
def user_scope(user_id):
    if not isinstance(user_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,160}', user_id):
        raise ValueError('Valid user identity required')
    token = _user.set(user_id)
    try:
        yield
    finally:
        _user.reset(token)
