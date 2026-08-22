from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings, repo_root


class Base(DeclarativeBase):
    pass


engine = None
SessionLocal: sessionmaker[Session] | None = None


def _sqlite_path(url: str) -> Path | None:
    prefix = "sqlite:///"
    if url == "sqlite:///:memory:" or url.startswith("sqlite+pysqlite:///:memory:"):
        return None
    if not url.startswith(prefix):
        return None
    raw = url[len(prefix) :]
    if raw == ":memory:":
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root() / path
    return path


def make_engine(database_url: str | None = None):
    url = database_url or get_settings().database_url
    path = _sqlite_path(url)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path}"
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, connect_args=connect_args)


def configure_db(database_url: str | None = None):
    global engine, SessionLocal
    engine = make_engine(database_url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return engine


def session_factory() -> sessionmaker[Session]:
    if SessionLocal is None:
        configure_db()
    assert SessionLocal is not None
    return SessionLocal
