"""
test_swarm_and_oracle.py
------------------------
Unit test suite for 9-Bot Swarm target configuration and Oracle Action Card Alert System.
"""

import json
import pytest
import httpx
from pathlib import Path
from app import app
from oracle_alerts import OracleActionCard, OracleAlertEngine, oracle_engine
from stochastic_drift import DriftAuditMetrics


def test_swarm_targets_configuration():
    target_file = Path("targets.json")
    assert target_file.exists()
    with target_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 9
    assert "betking_v_league" in data
    assert "betano_v_league" in data
    assert "msport_v_league" in data


def test_oracle_alert_engine_card_generation():
    engine = OracleAlertEngine()
    
    # Overextension metrics
    metrics = DriftAuditMetrics(
        timestamp=100.0,
        source="betking_v_league",
        rolling_rtp_10=1.45,
        rtp_mean=1.0,
        rtp_stddev=0.15,
        rtp_z_score=2.30,
        sequence_entropy=0.55,
        seed_volatility=1.90,
        drift_confidence=0.85,
        is_correction_phase=True,
        audit_message="PRNG Correction Phase ACTIVE",
    )

    card = engine.evaluate_metrics(metrics)
    assert card is not None
    assert card.target == "betking_v_league"
    assert card.actionable_play == "OVER_1_5_REBALANCE"
    assert card.condition == "PRNG_DEFENSIVE_DRAWDOWN"

    active_cards = engine.get_active_cards(limit=10)
    assert len(active_cards) == 1
    assert active_cards[0]["card_id"] == card.card_id


@pytest.mark.asyncio
async def test_oracle_cards_api_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/oracle-cards?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "cards" in data
