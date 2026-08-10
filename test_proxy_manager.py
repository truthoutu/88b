"""
test_proxy_manager.py
----------------------
Unit tests for proxy_manager.py (M2 architecture upgrade).
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from proxy_loader import ProxyConfig
from proxy_manager import (
    ProxyErrorKind,
    ProxyHealth,
    ProxyManager,
    RentalResult,
    classify_proxy_error,
)


def test_classify_proxy_error_taxonomy():
    """Verify error taxonomy mapping for various network and Playwright exceptions."""
    auth_exc = Exception("net::ERR_PROXY_AUTH_REQUESTED 407 Proxy Authentication Required")
    assert classify_proxy_error(auth_exc) == ProxyErrorKind.AUTH

    rate_exc = Exception("429 Too Many Requests")
    assert classify_proxy_error(rate_exc) == ProxyErrorKind.RATE_LIMIT

    handshake_exc = Exception("net::ERR_TUNNEL_CONNECTION_FAILED proxy_connection_failed")
    assert classify_proxy_error(handshake_exc) == ProxyErrorKind.PROXY_HANDSHAKE

    latency_exc = Exception("Navigation timeout of 30000 ms exceeded")
    assert classify_proxy_error(latency_exc) == ProxyErrorKind.NETWORK_LATENCY

    loss_exc = Exception("net::ERR_CONNECTION_RESET Connection reset by peer")
    assert classify_proxy_error(loss_exc) == ProxyErrorKind.PACKET_LOSS

    unknown_exc = Exception("Unexpected DOM error")
    assert classify_proxy_error(unknown_exc) == ProxyErrorKind.OTHER


@pytest.mark.asyncio
async def test_proxy_manager_direct_slot_lifecycle():
    """Verify renting and releasing direct slots (config=None)."""
    mgr = ProxyManager([None])
    assert mgr.summary().startswith("proxies 1")

    res = await mgr.rent_result()
    assert res.acquired is True
    assert res.config is None
    assert mgr.state_of(None) == "direct"

    # Second rent when in_use returns acquired=False
    res2 = await mgr.rent_result()
    assert res2.acquired is False

    # Release direct slot -> state returns to direct (healthy)
    await mgr.release(None, ok=True)
    assert mgr.state_of(None) == "direct"

    # Can rent again
    res3 = await mgr.rent_result()
    assert res3.acquired is True


@pytest.mark.asyncio
async def test_proxy_manager_circuit_breaker_and_quarantine():
    """Verify auth failure causes immediate quarantine, and repeated failures trip circuit breaker."""
    cfg1 = ProxyConfig(server="http://proxy1.example.com:8080", username=None, password=None, label="proxy1")
    cfg2 = ProxyConfig(server="http://proxy2.example.com:8080", username=None, password=None, label="proxy2")
    mgr = ProxyManager([cfg1, cfg2], circuit_threshold=3, quarantine_seconds=0.5, cooldown_base=0.1)

    # Rent cfg1
    p1 = await mgr.rent()
    assert p1 == cfg1

    # Auth error -> immediate quarantine
    await mgr.release(cfg1, ok=False, error=Exception("407 Proxy Authentication Required"))
    assert mgr.state_of(cfg1) == "quarantined"

    # Next rent gets cfg2
    p2 = await mgr.rent()
    assert p2 == cfg2

    # Fail cfg2 3 times (circuit threshold)
    await mgr.release(cfg2, ok=False, error=Exception("net::ERR_CONNECTION_RESET"))
    assert mgr.state_of(cfg2) == "cooling"

    # Rent again after cooldown
    await asyncio.sleep(0.15)
    p2_again = await mgr.rent()
    assert p2_again == cfg2
    await mgr.release(cfg2, ok=False, error=Exception("net::ERR_CONNECTION_RESET"))

    await asyncio.sleep(0.25)
    p2_again2 = await mgr.rent()
    assert p2_again2 == cfg2
    # 3rd failure trips circuit breaker -> quarantine
    await mgr.release(cfg2, ok=False, error=Exception("net::ERR_CONNECTION_RESET"))
    assert mgr.state_of(cfg2) == "quarantined"

    # Wait quarantine_seconds (0.5s) -> auto recover
    await asyncio.sleep(0.55)
    res_rec = await mgr.rent_result()
    assert res_rec.acquired is True
