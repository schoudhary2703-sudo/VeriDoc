"""Liveness endpoint.

Reports overall status plus per-dependency detail, so the officer dashboard can
show *what* is down, not merely that something is.
"""

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_redis
from app.config import get_settings
from app.db.session import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(
    db: AsyncSession = Depends(get_db),
    cache: aioredis.Redis = Depends(get_redis),
) -> dict:
    settings = get_settings()

    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:  # noqa: BLE001 - surface the reason to the dashboard
        db_status = f"error: {type(exc).__name__}"

    try:
        await cache.ping()
        redis_status = "ok"
    except Exception as exc:  # noqa: BLE001
        redis_status = f"error: {type(exc).__name__}"

    return {
        "status": "ok",
        "service": settings.app_name,
        "env": settings.env,
        "dependencies": {"database": db_status, "redis": redis_status},
    }
