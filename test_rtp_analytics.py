"""
test_rtp_analytics.py
---------------------
Unit test suite for RTP window calculator, mean reversion divergence triggers, and micro-probe validation endpoint.
"""

import pytest
import httpx
from app import app
from rtp_engine import RTPWindowCalculator, rtp_calculator


def test_rtp_window_calculator_initialization():
    calc = RTPWindowCalculator(divergence_threshold=0.15)
    stats = calc.get_rtp_stats("test_target")
    assert stats["short_rtp"] == 1.0
    assert stats["med_rtp"] == 1.0
    assert stats["macro_rtp"] == 1.0
    assert stats["divergence_delta"] == 0.0


def test_rtp_cycle_recording_and_divergence_signal():
    calc = RTPWindowCalculator(divergence_threshold=0.10, short_window=5, macro_window=10)

    for c in range(10):
        res = calc.record_cycle_outcome(
            source="test_platform",
            event_name=f"event_{c}",
            numeric_result=3.5,
            stake=100.0,
            cycle=c,
        )

    stats = calc.get_rtp_stats("test_platform")
    assert stats["sample_count"] == 10
    assert stats["short_rtp"] > 0.0


@pytest.mark.asyncio
async def test_micro_probe_api_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/probe", json={
            "target": "betking",
            "stake": 100.0,
            "predicted_odds": 2.10,
            "strategy": "VARIANCE_ARBITRAGE"
        })
        assert response.status_code == 200
        data = response.json()
        assert "probe_id" in data
        assert data["target"] == "betking"
        assert data["stake_ngn"] == 100.0
        assert data["execution_status"] == "EXECUTED_SUCCESS"
        assert "rtp_horizons" in data
