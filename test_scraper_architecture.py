"""
test_scraper_architecture.py
-----------------------------
Unit tests for scraper.py architectural helpers (M1 + M3).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from scraper import (
    _BLOCKED_RESOURCE_TYPES,
    _close_worker_context,
    _create_worker_context,
    _proxy_label,
    _random_fingerprint,
    _route_lighten,
)


@pytest.mark.asyncio
async def test_route_lighten_interception():
    """Verify M3 network lightening resource blocking logic."""
    # Test blocked resource types
    for res_type in ["image", "font", "media", "stylesheet", "imageset", "beacon", "csp_report"]:
        route = MagicMock()
        route.request.resource_type = res_type
        route.abort = AsyncMock()
        route.continue_ = AsyncMock()

        await _route_lighten(route)

        route.abort.assert_awaited_once()
        route.continue_.assert_not_called()

    # Test allowed resource types
    for res_type in ["document", "xhr", "fetch", "script"]:
        route = MagicMock()
        route.request.resource_type = res_type
        route.abort = AsyncMock()
        route.continue_ = AsyncMock()

        await _route_lighten(route)

        route.continue_.assert_awaited_once()
        route.abort.assert_not_called()


def test_random_fingerprint_generation():
    """Verify fingerprint randomization properties."""
    fp1 = _random_fingerprint(seed=42)
    fp2 = _random_fingerprint(seed=42)
    assert fp1 == fp2  # Deterministic with same seed

    fp3 = _random_fingerprint()
    assert "viewport" in fp3
    assert "user_agent" in fp3
    assert "locale" in fp3
    assert "timezone_id" in fp3
    assert "device_scale_factor" in fp3


@pytest.mark.asyncio
async def test_create_and_close_worker_context():
    """Verify M1 context pool semaphore acquisition and cleanup."""
    semaphore = asyncio.Semaphore(1)
    mock_browser = MagicMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()

    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_context.new_page = AsyncMock(return_value=mock_page)

    ctx, page = await _create_worker_context(mock_browser, proxy=None, semaphore=semaphore)
    assert ctx == mock_context
    assert page == mock_page
    assert semaphore.locked() is True

    # Close context releases semaphore slot
    await _close_worker_context(ctx, semaphore)
    assert semaphore.locked() is False
    mock_context.close.assert_awaited_once()
