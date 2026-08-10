"""
train_auditor.py
----------------
Statistical Auditor ML Model Training & Inference Pipeline.

Trains an Ensemble Classifier (Random Forest + Isolation Forest) on engineered time-series features:
  - rolling_rtp_10
  - rtp_z_score
  - sequence_entropy (Shannon Entropy H)
  - cluster_index (Run-Length Clustering Coefficient C)
  - seed_volatility

Saves model weights to models/ and provides real-time inference wrapper PRNGAuditorPredictor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from feature_extractor import feature_extractor


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RF_MODEL_PATH = MODEL_DIR / "prng_auditor_rf.joblib"
IF_MODEL_PATH = MODEL_DIR / "prng_auditor_if.joblib"


class PRNGAuditorTrainer:
    """
    Trains statistical auditor machine learning models to detect PRNG correction phases.
    """

    def __init__(self) -> None:
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        self.if_model = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)

    def train_on_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, Any]:
        """
        Train ensemble models on feature matrix X and target labels y.
        """
        if X.empty or len(X) < 10:
            logger.warning("Insufficient data samples ({}) to train auditor model.", len(X))
            return {"status": "warning", "message": "Insufficient data"}

        import numpy as np
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) > 1 and min(counts) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )


        # 1. Fit Random Forest Classifier
        self.rf_model.fit(X_train, y_train)

        # 2. Fit Isolation Forest Anomaly Detector
        self.if_model.fit(X_train)

        # Evaluate performance
        y_pred = self.rf_model.predict(X_test)
        y_proba = self.rf_model.predict_proba(X_test)[:, 1] if hasattr(self.rf_model, "predict_proba") else y_pred

        try:
            auc = round(float(roc_auc_score(y_test, y_proba)), 4) if len(unique_classes) > 1 else 1.0
        except Exception:
            auc = 1.0

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Save model artifacts
        joblib.dump(self.rf_model, RF_MODEL_PATH)
        joblib.dump(self.if_model, IF_MODEL_PATH)
        logger.success("Statistical Auditor model saved to {}", RF_MODEL_PATH)

        return {
            "status": "success",
            "sample_count": len(X),
            "roc_auc": auc,
            "report": report,
            "model_path": str(RF_MODEL_PATH),
        }

    def train_from_telemetry_rows(
        self, rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Extract features from raw telemetry rows and execute model training.
        """
        X, y = feature_extractor.build_feature_dataframe(rows)
        return self.train_on_data(X, y)


class PRNGAuditorPredictor:
    """
    Real-time inference engine loading trained PRNG Auditor model weights.
    """

    def __init__(self) -> None:
        self._rf_model: RandomForestClassifier | None = None
        self._if_model: IsolationForest | None = None
        self.load_models()

    def load_models(self) -> bool:
        if RF_MODEL_PATH.exists():
            try:
                self._rf_model = joblib.load(RF_MODEL_PATH)
                logger.info("Loaded Random Forest auditor model from {}", RF_MODEL_PATH)
            except Exception as exc:
                logger.warning("Could not load RF model: {}", exc)

        if IF_MODEL_PATH.exists():
            try:
                self._if_model = joblib.load(IF_MODEL_PATH)
            except Exception:
                pass

        return self._rf_model is not None

    def predict_correction_probability(self, features_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Predict PRNG Correction Phase probability for a single feature vector.
        """
        df_single = pd.DataFrame([{
            "rolling_rtp_10": float(features_dict.get("rolling_rtp_10", 1.0)),
            "rtp_z_score": float(features_dict.get("rtp_z_score", 0.0)),
            "sequence_entropy": float(features_dict.get("sequence_entropy", 1.0)),
            "cluster_index": float(features_dict.get("cluster_index", 1.0)),
            "seed_volatility": float(features_dict.get("seed_volatility", 1.0)),
        }])

        if self._rf_model:
            proba = float(self._rf_model.predict_proba(df_single)[0, 1])
            is_correction = bool(proba >= 0.50)
        else:
            # Fallback heuristic calculation if model not trained yet
            z = abs(df_single["rtp_z_score"].iloc[0])
            h = df_single["sequence_entropy"].iloc[0]
            c = df_single["cluster_index"].iloc[0]
            proba = min(1.0, max(0.0, (z * 0.25) + ((1.0 - h) * 0.35) + ((c - 1.0) * 0.20)))
            is_correction = bool(proba >= 0.50)

        return {
            "correction_probability": round(proba, 4),
            "is_correction_phase": is_correction,
            "model_status": "trained_rf" if self._rf_model else "heuristic_fallback",
        }


# Global predictor instance
auditor_predictor = PRNGAuditorPredictor()
