from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def get_engine():
    return create_engine(get_settings().sqlalchemy_url, pool_pre_ping=True)


@lru_cache
def _session_factory():
    return sessionmaker(get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Session:
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
