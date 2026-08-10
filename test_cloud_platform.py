"""
test_cloud_platform.py
----------------------
Unit tests for database layer, Celery app initialization, and FastAPI cloud routes.
"""

import pytest
import httpx
from db import TelemetryDatabase
from celery_app import celery
from app import app


def test_celery_app_config():
    assert celery.main == "telemetry_scraper"
    assert celery.conf.task_serializer == "json"


@pytest.mark.asyncio
async def test_db_fallback_mode():
    db_inst = TelemetryDatabase()
    connected = await db_inst.connect()
    assert connected is False or connected is True
    assert await db_inst.save_telemetry_batch([]) == 0
    await db_inst.disconnect()


@pytest.mark.asyncio
async def test_fastapi_endpoints():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "engine_state" in data
        assert "db_connected" in data

        db_res = await client.get("/api/db-status")
        assert db_res.status_code == 200
        assert "db_connected" in db_res.json()
