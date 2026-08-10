"""
test_feature_extractor_and_trainer.py
--------------------------------------
Unit test suite for Feature Extractor (Entropy, Cluster Index) and Statistical Auditor ML Pipeline.
"""

import pytest
import httpx
import pandas as pd
from app import app
from feature_extractor import (
    compute_sequence_entropy,
    compute_cluster_index,
    feature_extractor,
)
from train_auditor import PRNGAuditorTrainer, auditor_predictor


def test_compute_sequence_entropy():
    # Deterministic sequence (H == 0)
    assert compute_sequence_entropy([1, 1, 1, 1, 1]) == 0.0
    # Balanced binary sequence (H ~ 1.0)
    assert compute_sequence_entropy([1, 0, 1, 0, 1, 0]) > 0.95


def test_compute_cluster_index():
    # Alternating sequence (Low run lengths)
    c_low = compute_cluster_index([1, 0, 1, 0, 1, 0])
    assert c_low <= 1.0

    # Heavy run length clustering (High Cluster Index C > 1.5)
    c_high = compute_cluster_index([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert c_high > 1.5


def test_feature_extractor_dataframe():
    mock_rows = [
        {"timestamp": "1", "source": "betking", "cycle": i, "numeric_result": "2.0"}
        for i in range(15)
    ]
    X, y = feature_extractor.build_feature_dataframe(mock_rows)
    assert not X.empty
    assert len(X) == 15
    assert "sequence_entropy" in X.columns
    assert "cluster_index" in X.columns


def test_prng_auditor_trainer_and_predictor():
    mock_rows = [
        {"timestamp": str(i), "source": "betano", "cycle": i, "numeric_result": "2.10"}
        for i in range(30)
    ]
    trainer = PRNGAuditorTrainer()
    res = trainer.train_from_telemetry_rows(mock_rows)
    assert res["status"] == "success"
    assert "model_path" in res

    # Test inference predictor
    pred = auditor_predictor.predict_correction_probability({
        "rolling_rtp_10": 1.50,
        "rtp_z_score": 2.50,
        "sequence_entropy": 0.40,
        "cluster_index": 2.10,
        "seed_volatility": 1.90,
    })
    assert "correction_probability" in pred
    assert "is_correction_phase" in pred


@pytest.mark.asyncio
async def test_audit_predict_api_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/audit-predict", json={
            "rolling_rtp_10": 1.45,
            "rtp_z_score": 2.20,
            "sequence_entropy": 0.50,
            "cluster_index": 1.80,
            "seed_volatility": 1.95,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "prediction" in data
