"""
feature_extractor.py
--------------------
Feature Extraction Module for Stochastic Drift & PRNG Statistical Auditor.

Calculates:
  1. Sequence Entropy (Shannon Entropy H):
     H = - sum( p_i * log2(p_i) ) over outcome distribution.

  2. Cluster Index (Run-Length Clustering Coefficient C):
     C = Mean_Observed_Run_Length / Theoretical_Random_Run_Length
     Measures artificial outcome clustering and false streaks.

  3. Feature Matrix Builder:
     Extracts structured Pandas DataFrames from Supabase vault time-series records.
"""

from __future__ import annotations

import math
import statistics
from typing import Any
import pandas as pd
from loguru import logger


def compute_sequence_entropy(outcomes: list[int]) -> float:
    """
    Calculate Shannon Entropy (bits) over binary outcome sequence.
    Formula: H = - sum( p_i * log2(p_i) )
    Max entropy for binary outcome = 1.0 bit. Zero entropy = 0.0 (deterministic streak).
    """
    if not outcomes:
        return 1.0

    n = len(outcomes)
    p1 = sum(outcomes) / n
    p0 = 1.0 - p1

    if p0 <= 0.0 or p1 <= 0.0:
        return 0.0

    entropy = - (p0 * math.log2(p0) + p1 * math.log2(p1))
    return round(entropy, 4)


def compute_cluster_index(outcomes: list[int]) -> float:
    """
    Calculate Run-Length Clustering Coefficient (C).
    Formula: C = Mean_Observed_Run_Length / Theoretical_Random_Run_Length

    Runs are contiguous sequences of identical outcomes (e.g. [1, 1, 1, 0, 0, 1]).
    For fair 50/50 Bernoulli trials, theoretical mean run length = 2.0.
    A Cluster Index C > 1.5 indicates heavy outcome clustering / false streak manipulation.
    """
    if not outcomes or len(outcomes) < 2:
        return 1.0

    runs = []
    current_len = 1
    for i in range(1, len(outcomes)):
        if outcomes[i] == outcomes[i - 1]:
            current_len += 1
        else:
            runs.append(current_len)
            current_len = 1
    runs.append(current_len)

    mean_observed_run = statistics.mean(runs)
    p1 = max(0.05, min(0.95, sum(outcomes) / len(outcomes)))
    theo_mean_run = 1.0 / (1.0 - 2.0 * p1 * (1.0 - p1))

    if theo_mean_run <= 0.0:
        theo_mean_run = 2.0

    cluster_index = mean_observed_run / theo_mean_run
    return round(cluster_index, 4)


class StochasticFeatureExtractor:
    """
    Extracts engineered feature vectors and Pandas DataFrames from raw Supabase vault time-series rows.
    """

    FEATURE_COLUMNS = [
        "rolling_rtp_10",
        "rtp_z_score",
        "sequence_entropy",
        "cluster_index",
        "seed_volatility",
    ]

    def extract_features_from_records(
        self,
        rows: list[dict[str, Any]],
        window_size: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Processes time-series rows to compute sliding Sequence Entropy, Cluster Index, and Z-Score features.
        """
        if not rows:
            return []

        # Sort chronologically
        sorted_rows = sorted(rows, key=lambda r: r.get("id", 0) or r.get("timestamp", ""))
        extracted_features = []

        outcomes_history = []
        payouts_history = []
        stakes_history = []
        odds_history = []
        rtp_10_history = []

        for r in sorted_rows:
            numeric_val = r.get("numeric_result", 1.95)
            try:
                odds = float(numeric_val) if numeric_val else 1.95
                if odds <= 1.0:
                    odds = 1.95
            except (ValueError, TypeError):
                odds = 1.95

            cycle = int(r.get("cycle", 0))
            won = (cycle % 2 == 0)
            stake = float(r.get("synthetic_stake", 100.0))
            payout = float(r.get("actual_payout", stake * odds if won else 0.0))

            outcomes_history.append(1 if won else 0)
            payouts_history.append(payout)
            stakes_history.append(stake)
            odds_history.append(odds)

            # Rolling 10-cycle RTP
            sub_payouts = payouts_history[-10:]
            sub_stakes = stakes_history[-10:]
            rolling_rtp_10 = round(sum(sub_payouts) / max(1.0, sum(sub_stakes)), 4)
            rtp_10_history.append(rolling_rtp_10)

            # Sequence Entropy & Cluster Index over sliding window
            window_outcomes = outcomes_history[-window_size:]
            entropy = compute_sequence_entropy(window_outcomes)
            cluster_idx = compute_cluster_index(window_outcomes)

            # Z-Score
            if len(rtp_10_history) >= 3:
                mean_rtp = statistics.mean(rtp_10_history)
                try:
                    std_rtp = statistics.stdev(rtp_10_history)
                except statistics.StatisticsError:
                    std_rtp = 0.05
                if std_rtp < 0.001:
                    std_rtp = 0.01
                z_score = round((rolling_rtp_10 - mean_rtp) / std_rtp, 4)
            else:
                z_score = 0.0

            # Seed Volatility
            if len(payouts_history) >= 3:
                try:
                    obs_std = statistics.stdev(payouts_history[-30:])
                except statistics.StatisticsError:
                    obs_std = 0.0
                mean_odds = statistics.mean(odds_history[-30:])
                p_win = min(0.95, max(0.05, 1.0 / mean_odds))
                theo_std = math.sqrt(p_win * (1.0 - p_win)) * (stake * mean_odds)
                seed_vol = round(obs_std / theo_std, 4) if theo_std > 0 else 1.0
            else:
                seed_vol = 1.0

            # Ground truth correction phase label (Z >= 2.0 or H < 0.65 or Cluster > 1.5)
            is_correction = bool(abs(z_score) >= 2.0 or entropy <= 0.65 or cluster_idx >= 1.5)

            feature_dict = {
                "timestamp": str(r.get("timestamp", "")),
                "source": str(r.get("source", "unknown")),
                "rolling_rtp_10": rolling_rtp_10,
                "rtp_z_score": z_score,
                "sequence_entropy": entropy,
                "cluster_index": cluster_idx,
                "seed_volatility": seed_vol,
                "is_correction_phase": int(is_correction),
            }
            extracted_features.append(feature_dict)

        return extracted_features

    def build_feature_dataframe(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Construct feature matrix X (DataFrame) and target labels y (Series) for ML training.
        """
        features_list = self.extract_features_from_records(rows)
        if not features_list:
            df_empty = pd.DataFrame(columns=self.FEATURE_COLUMNS)
            return df_empty, pd.Series(dtype=int)

        df = pd.DataFrame(features_list)
        X = df[self.FEATURE_COLUMNS]
        y = df["is_correction_phase"]
        return X, y


# Global singleton instance
feature_extractor = StochasticFeatureExtractor()
