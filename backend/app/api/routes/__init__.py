from fastapi import APIRouter

from app.api.routes import audit, health, verify

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(verify.router)
api_router.include_router(audit.router)
