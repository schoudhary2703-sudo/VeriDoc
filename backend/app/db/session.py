"""Async SQLAlchemy engine and session factory.

The engine is created lazily and falls back to SQLite when the configured
driver is unavailable. That matters for the demo: the verification pipeline and
the audit log do not need Postgres, so a missing database driver should cost the
health endpoint one degraded line, not prevent the API from starting at all.

In Docker, `DATABASE_URL` points at Postgres and this fallback never fires.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

logger = logging.getLogger(__name__)

# Local file database used when the configured driver is not installed.
SQLITE_FALLBACK_URL = "sqlite+aiosqlite:///./veridoc-local.db"


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Create the engine on first use, falling back to SQLite if needed."""
    settings = get_settings()
    try:
        return create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
    except Exception as exc:  # noqa: BLE001 - missing driver, bad URL, etc.
        logger.warning(
            "could not create an engine for the configured DATABASE_URL (%s); "
            "falling back to %s",
            type(exc).__name__,
            SQLITE_FALLBACK_URL,
        )
        return create_async_engine(SQLITE_FALLBACK_URL, echo=False)


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with get_session_factory()() as session:
        yield session
