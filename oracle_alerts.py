"""
oracle_alerts.py
----------------
Oracle Action Card Alert System.

Translates PRNG stochastic divergence, Z-score overextensions, and Shannon entropy drops
into clean, human-readable advisory cards for the Flight Deck UI and FastAPI control plane.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, asdict
from typing import Any
from loguru import logger


@dataclass
class OracleActionCard:
    card_id: str
    target: str
    condition: str
    active_window: str
    confidence_score: float
    actionable_play: str
    description: str
    timestamp: float


class OracleAlertEngine:
    """
    Evaluates stochastic audit metrics and generates real-time Oracle Action Cards.
    """

    def __init__(self, card_history_limit: int = 50) -> None:
        self.card_history_limit = card_history_limit
        self._cards: collections.deque[OracleActionCard] = collections.deque(maxlen=self.card_history_limit)

    def evaluate_metrics(self, drift_metrics: Any) -> OracleActionCard | None:
        """
        Convert DriftAuditMetrics into an actionable Oracle Advisory Card.
        """
        if not drift_metrics:
            return None

        target = getattr(drift_metrics, "source", "unknown")
        z_score = getattr(drift_metrics, "rtp_z_score", 0.0)
        entropy = getattr(drift_metrics, "sequence_entropy", 1.0)
        seed_vol = getattr(drift_metrics, "seed_volatility", 1.0)
        confidence = getattr(drift_metrics, "drift_confidence", 0.0)
        is_correction = getattr(drift_metrics, "is_correction_phase", False)

        if not is_correction and abs(z_score) < 1.5 and entropy >= 0.70:
            return None

        # Determine condition and actionable play
        card_id = f"CARD-{int(time.time() * 1000) % 1000000:06d}"
        active_window = "10-Cycle Short Horizon"

        if z_score >= 2.0:
            condition = "PRNG_DEFENSIVE_DRAWDOWN"
            actionable_play = "OVER_1_5_REBALANCE"
            desc = f"PRNG overextended (Z = {z_score:+.2f}). Mathematical pullback phase active."
        elif z_score <= -2.0:
            condition = "PRNG_EXPANSION_SPIKE"
            actionable_play = "UNDER_2_5_CORRECTION"
            desc = f"PRNG depressed (Z = {z_score:+.2f}). Upward payout release imminent."
        elif entropy <= 0.65:
            condition = "ARTIFICIAL_CLUSTERING_DETECTED"
            actionable_play = "STREAK_BREAK_REBALANCE"
            desc = f"Low sequence entropy (H = {entropy:.2f}). Artificial outcome clustering flagged."
        elif seed_vol >= 1.8:
            condition = "SEED_PARAMETER_SHIFT"
            actionable_play = "HIGH_VARIANCE_PASS"
            desc = f"Seed volatility spike ({seed_vol:.2f}x). Market parameter shift in progress."
        else:
            condition = "MODERATE_PRNG_DRIFT"
            actionable_play = "MONITOR_STOCHASTIC_BOUNDS"
            desc = f"Elevated drift confidence ({(confidence*100):.1f}%)."

        card = OracleActionCard(
            card_id=card_id,
            target=target,
            condition=condition,
            active_window=active_window,
            confidence_score=round(confidence, 4),
            actionable_play=actionable_play,
            description=desc,
            timestamp=time.time(),
        )

        self._cards.appendleft(card)
        logger.info("[ORACLE CARD] Generated advisory {} for target={}: {} -> {}",
                    card_id, target, condition, actionable_play)
        return card

    def get_active_cards(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Return current active Oracle Action Cards as list of dictionaries.
        """
        return [asdict(card) for card in list(self._cards)[:limit]]


# Global singleton instance
oracle_engine = OracleAlertEngine()
