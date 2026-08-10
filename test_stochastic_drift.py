"""
test_stochastic_drift.py
------------------------
Unit test suite for Stochastic Drift Auditor (Z-Score, Shannon Entropy, Seed Volatility)
and Supabase feature store endpoints.
"""

import pytest
import httpx
from app import app
from stochastic_drift import StochasticDriftAuditor, drift_auditor


def test_stochastic_drift_auditor_initialization():
    auditor = StochasticDriftAuditor(z_score_threshold=2.0, entropy_threshold=0.65)
    metrics = auditor.audit_cycle("test_source", current_rtp_10=1.0, won=True, actual_payout=200.0)
    assert metrics.source == "test_source"
    assert metrics.rolling_rtp_10 == 1.0
    assert metrics.sequence_entropy >= 0.0
    assert metrics.seed_volatility >= 0.0


def test_shannon_entropy_calculation():
    auditor = StochasticDriftAuditor()
    # Uniform alternating sequence (H ~ 1.0)
    h_uniform = auditor._compute_sequence_entropy([1, 0, 1, 0, 1, 0, 1, 0])
    assert h_uniform > 0.95

    # Completely deterministic sequence (H == 0.0)
    h_det = auditor._compute_sequence_entropy([1, 1, 1, 1, 1, 1, 1, 1])
    assert h_det == 0.0


def test_z_score_overextension_trigger():
    auditor = StochasticDriftAuditor(z_score_threshold=1.5)
    
    # Feed baseline RTP values
    for _ in range(10):
        auditor.audit_cycle("overextend_target", current_rtp_10=1.0, won=False, actual_payout=0.0)

    # Spike RTP significantly to trigger positive Z-Score
    m_spike = auditor.audit_cycle("overextend_target", current_rtp_10=2.5, won=True, actual_payout=500.0)
    assert m_spike.rtp_z_score > 1.0


@pytest.mark.asyncio
async def test_drift_features_api_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/drift-features?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "features" in data
