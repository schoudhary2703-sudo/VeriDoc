"""Phase 0 smoke tests.

These run without Postgres or Redis: the health endpoint degrades to reporting
per-dependency errors rather than failing, so the API contract stays testable in
isolation.
"""

import httpx
import pytest

from app.main import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_root(client: httpx.AsyncClient) -> None:
    resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "VeriDoc"


async def test_health_shape(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["dependencies"]) == {"database", "redis"}
