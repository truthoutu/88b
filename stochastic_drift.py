"""
stochastic_drift.py
-------------------
Stochastic Drift Detection & PRNG Statistical Auditor Engine.

Core Objective Function:
  Abandon outcome prediction. Audit closed-loop PRNG mathematical rebalancing cycles
  and detect forced correction phases.

Core Metrics:
  1. RTP Deviation Score (Z-Score):
     Z = (RTP_10 - Mean_RTP) / StdDev_RTP
     Flags overextended PRNG state when |Z| >= 2.0.

  2. Sequence Entropy Metric (Shannon Entropy):
     H(S) = - sum( p_i * log2(p_i) )
     Measures outcome randomness. Entropy H < 0.65 flags artificial outcome clustering / false streaks.

  3. Seed Volatility Index:
     Vol_seed = StdDev(Observed_Payouts) / StdDev(Theoretical_Payouts)
     Detects seed parameter shifts in closed-loop PRNG.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any
from loguru import logger


@dataclass
class DriftAuditMetrics:
    timestamp: float
    source: str
    rolling_rtp_10: float
    rtp_mean: float
    rtp_stddev: float
    rtp_z_score: float
    sequence_entropy: float
    seed_volatility: float
    drift_confidence: float
    is_correction_phase: bool
    audit_message: str


class StochasticDriftAuditor:
    """
    Statistical Auditor hunting PRNG mathematical correction phases.
    Processes cycle outcomes to compute Z-Score, Shannon Entropy, and Seed Volatility.
    """

    def __init__(
        self,
        z_score_threshold: float = 2.0,
        entropy_threshold: float = 0.65,
        volatility_threshold: float = 1.8,
        history_limit: int = 200,
    ) -> None:
        self.z_score_threshold = z_score_threshold
        self.entropy_threshold = entropy_threshold
        self.volatility_threshold = volatility_threshold
        self.history_limit = history_limit

        # Map source -> list of payout/stake historical records
        self._rtp_history: dict[str, list[float]] = {}
        self._outcome_history: dict[str, list[int]] = {}  # 1 for win, 0 for loss
        self._payout_history: dict[str, list[float]] = {}
        self._odds_history: dict[str, list[float]] = {}

    def audit_cycle(
        self,
        source: str,
        current_rtp_10: float,
        won: bool,
        actual_payout: float,
        stake: float = 100.0,
        odds: float = 1.95,
    ) -> DriftAuditMetrics:
        """
        Audit a new cycle result, compute stochastic drift metrics, and evaluate correction phase.
        """
        if source not in self._rtp_history:
            self._rtp_history[source] = []
            self._outcome_history[source] = []
            self._payout_history[source] = []
            self._odds_history[source] = []

        rtp_list = self._rtp_history[source]
        outcome_list = self._outcome_history[source]
        payout_list = self._payout_history[source]
        odds_list = self._odds_history[source]

        rtp_list.append(current_rtp_10)
        outcome_list.append(1 if won else 0)
        payout_list.append(actual_payout)
        odds_list.append(odds)

        if len(rtp_list) > self.history_limit:
            rtp_list.pop(0)
            outcome_list.pop(0)
            payout_list.pop(0)
            odds_list.pop(0)

        # 1. Compute Z-Score
        z_score, mean_rtp, std_rtp = self._compute_rtp_zscore(rtp_list, current_rtp_10)

        # 2. Compute Sequence Entropy (Shannon Entropy)
        entropy = self._compute_sequence_entropy(outcome_list[-20:])

        # 3. Compute Seed Volatility Index
        seed_vol = self._compute_seed_volatility(payout_list[-30:], odds_list[-30:], stake)

        # 4. Evaluate Correction Phase Trigger
        is_correction, confidence, msg = self._evaluate_correction_phase(
            z_score, entropy, seed_vol
        )

        metrics = DriftAuditMetrics(
            timestamp=time.time(),
            source=source,
            rolling_rtp_10=round(current_rtp_10, 4),
            rtp_mean=round(mean_rtp, 4),
            rtp_stddev=round(std_rtp, 4),
            rtp_z_score=round(z_score, 4),
            sequence_entropy=round(entropy, 4),
            seed_volatility=round(seed_vol, 4),
            drift_confidence=round(confidence, 4),
            is_correction_phase=is_correction,
            audit_message=msg,
        )

        if is_correction:
            logger.warning("[PRNG CORRECTION PHASE] Source={}: Z={:.2f}, H={:.2f}, Vol={:.2f} -> {}",
                           source, z_score, entropy, seed_vol, msg)

        return metrics

    def _compute_rtp_zscore(
        self, rtp_series: list[float], current_rtp: float
    ) -> tuple[float, float, float]:
        if len(rtp_series) < 3:
            return 0.0, current_rtp, 0.05

        mean_val = statistics.mean(rtp_series)
        try:
            stdev_val = statistics.stdev(rtp_series)
        except statistics.StatisticsError:
            stdev_val = 0.05

        if stdev_val < 0.001:
            stdev_val = 0.01

        z_score = (current_rtp - mean_val) / stdev_val
        return z_score, mean_val, stdev_val

    def _compute_sequence_entropy(self, window: list[int]) -> float:
        """
        Calculate Shannon Entropy over sliding outcome sequence.
        H(S) = - sum( p_i * log2(p_i) )
        Max entropy for binary coin flip = 1.0 bit.
        """
        if not window:
            return 1.0

        n = len(window)
        p1 = sum(window) / n
        p0 = 1.0 - p1

        if p0 == 0.0 or p1 == 0.0:
            return 0.0  # Zero randomness (completely deterministic pattern)

        entropy = - (p0 * math.log2(p0) + p1 * math.log2(p1))
        return round(entropy, 4)

    def _compute_seed_volatility(
        self, payouts: list[float], odds_list: list[float], stake: float
    ) -> float:
        """
        Seed Volatility = Observed Payout StdDev / Theoretical Payout StdDev
        """
        if len(payouts) < 3:
            return 1.0

        try:
            obs_std = statistics.stdev(payouts)
        except statistics.StatisticsError:
            obs_std = 0.0

        # Theoretical variance for binary Bernoulli outcome at mean odds
        mean_odds = statistics.mean(odds_list) if odds_list else 1.95
        p_win = min(0.95, max(0.05, 1.0 / mean_odds))
        theo_std = math.sqrt(p_win * (1.0 - p_win)) * (stake * mean_odds)

        if theo_std == 0:
            return 1.0

        ratio = obs_std / theo_std
        return round(ratio, 4)

    def _evaluate_correction_phase(
        self, z_score: float, entropy: float, seed_vol: float
    ) -> tuple[bool, float, str]:
        """
        Determine if PRNG is in a mathematical correction phase based on drift metrics.
        """
        is_z_exceeded = abs(z_score) >= self.z_score_threshold
        is_entropy_low = entropy <= self.entropy_threshold
        is_vol_high = seed_vol >= self.volatility_threshold

        score = 0.0
        reasons = []

        if is_z_exceeded:
            score += 0.45
            reasons.append(f"Z-Score ({z_score:+.2f}) overextended")
        if is_entropy_low:
            score += 0.35
            reasons.append(f"Low Entropy ({entropy:.2f}) artificial clustering")
        if is_vol_high:
            score += 0.20
            reasons.append(f"Seed Volatility ({seed_vol:.2f}) parameter shift")

        confidence = round(min(1.0, score), 4)
        is_correction = confidence >= 0.50

        if is_correction:
            msg = "PRNG Correction Phase ACTIVE: " + "; ".join(reasons)
        else:
            msg = "PRNG Equilibrium: Metrics within normal stochastic bounds."

        return is_correction, confidence, msg


# Global singleton instance
drift_auditor = StochasticDriftAuditor()
