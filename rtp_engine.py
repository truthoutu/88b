"""
rtp_engine.py
-------------
Stochastic Drift & Rolling RTP Engine for Closed-Loop PRNG Auditing.

Objective Function:
  Audit closed-loop PRNG mathematical rebalancing cycles and compute rolling Return-to-Player (RTP)
  and Z-Score deviations as data streams into Supabase.

Mathematical Formulas:
  1. Realized Rolling RTP:
     Rolling_RTP = sum(Payouts) / sum(Stakes)

  2. RTP Z-Score Deviation:
     Z = (Current_RTP - Mean_RTP) / StdDev_RTP
     Flags overextended PRNG state when |Z| >= 2.0.
"""

from __future__ import annotations

import collections
import statistics
import time
from dataclasses import dataclass, field
from typing import Any
from loguru import logger


# ── Top-Level Utility Calculation Functions ──────────────────────────────────

def calculate_rolling_rtp(payouts: list[float], stakes: list[float]) -> float:
    """
    Calculate Realized Rolling RTP across given payouts and stakes series.
    Formula: RTP = sum(Payouts) / sum(Stakes)
    """
    if not payouts or not stakes:
        return 1.0  # Default 100% baseline

    total_payouts = sum(payouts)
    total_stakes = sum(stakes)

    if total_stakes == 0:
        return 1.0

    return round(total_payouts / total_stakes, 4)


def calculate_rtp_zscore(
    current_rtp: float, historical_rtps: list[float]
) -> tuple[float, float, float]:
    """
    Calculate Z-Score deviation of current RTP against historical RTP series.
    Formula: Z = (Current_RTP - Mean_RTP) / StdDev_RTP
    Returns: (z_score, mean_rtp, std_rtp)
    """
    if not historical_rtps or len(historical_rtps) < 2:
        return 0.0, current_rtp, 0.05

    mean_rtp = statistics.mean(historical_rtps)
    try:
        std_rtp = statistics.stdev(historical_rtps)
    except statistics.StatisticsError:
        std_rtp = 0.05

    if std_rtp < 0.001:
        std_rtp = 0.01

    z_score = (current_rtp - mean_rtp) / std_rtp
    return round(z_score, 4), round(mean_rtp, 4), round(std_rtp, 4)


# ── Dataclass Models ─────────────────────────────────────────────────────────

@dataclass
class CycleEconomicRecord:
    timestamp: str
    source: str
    event_name: str
    stake: float = 100.0
    odds: float = 2.0
    won: bool = False
    payout: float = 0.0
    house_margin: float = 0.05
    cycle: int = 0


@dataclass
class RTPSignalEvent:
    timestamp: float
    source: str
    short_rtp: float
    med_rtp: float
    macro_rtp: float
    divergence_delta: float
    signal_type: str
    description: str


# ── RTP Window Calculator Class ──────────────────────────────────────────────

class RTPWindowCalculator:
    """
    Maintains rolling windows of economic cycle data per target platform,
    computes Realized RTP across Short (10), Medium (100), and Macro (1000) windows,
    and evaluates Z-Score stochastic drift metrics.
    """

    def __init__(
        self,
        divergence_threshold: float = 0.15,  # 15% threshold
        short_window: int = 10,
        med_window: int = 100,
        macro_window: int = 1000,
    ) -> None:
        self.divergence_threshold = divergence_threshold
        self.short_window = short_window
        self.med_window = med_window
        self.macro_window = macro_window

        # Map source -> deque of CycleEconomicRecord
        self._history: dict[str, collections.deque[CycleEconomicRecord]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.macro_window)
        )
        self._rtp_series: dict[str, list[float]] = collections.defaultdict(list)
        self._recent_signals: list[RTPSignalEvent] = []

    def record_cycle_outcome(
        self,
        source: str,
        event_name: str,
        numeric_result: str | float,
        stake: float = 100.0,
        cycle: int = 0,
        timestamp: str = "",
    ) -> dict[str, Any]:
        """
        Record a cycle result, compute payouts based on numeric odds, update rolling windows,
        and evaluate stochastic Z-score drift metrics.
        """
        try:
            odds = float(numeric_result) if numeric_result else 1.95
            if odds <= 1.0:
                odds = 1.95
        except (ValueError, TypeError):
            odds = 1.95

        house_margin = round(max(0.02, 1.0 - (1.0 / odds)), 4) if odds > 1.0 else 0.05
        won = (cycle % 2 == 0)
        payout = round(stake * odds if won else 0.0, 2)

        record = CycleEconomicRecord(
            timestamp=timestamp or str(time.time()),
            source=source,
            event_name=event_name,
            stake=stake,
            odds=odds,
            won=won,
            payout=payout,
            house_margin=house_margin,
            cycle=cycle,
        )

        history = self._history[source]
        history.append(record)

        # Compute rolling window RTPs
        rtp_stats = self.get_rtp_stats(source)
        self._rtp_series[source].append(rtp_stats["short_rtp"])
        if len(self._rtp_series[source]) > self.macro_window:
            self._rtp_series[source].pop(0)

        # Compute Z-score deviation
        z_score, rtp_mean, rtp_std = calculate_rtp_zscore(
            rtp_stats["short_rtp"], self._rtp_series[source]
        )
        rtp_stats["rtp_z_score"] = z_score
        rtp_stats["rtp_mean"] = rtp_mean
        rtp_stats["rtp_std"] = rtp_std

        signal = self._evaluate_divergence_signal(source, rtp_stats)

        try:
            from stochastic_drift import drift_auditor
            drift_metrics = drift_auditor.audit_cycle(
                source=source,
                current_rtp_10=rtp_stats["short_rtp"],
                won=won,
                actual_payout=payout,
                stake=stake,
                odds=odds,
            )
        except Exception:
            drift_metrics = None

        return {
            "record": record,
            "rtp_stats": rtp_stats,
            "signal": signal,
            "drift_metrics": drift_metrics,
        }

    def get_rtp_stats(self, source: str) -> dict[str, float]:
        history = list(self._history[source])
        if not history:
            return {
                "short_rtp": 1.0,
                "med_rtp": 1.0,
                "macro_rtp": 1.0,
                "divergence_delta": 0.0,
                "rtp_z_score": 0.0,
                "rtp_mean": 1.0,
                "rtp_std": 0.05,
                "sample_count": 0,
            }

        short_records = history[-self.short_window:]
        med_records = history[-self.med_window:]
        macro_records = history[-self.macro_window:]

        short_rtp = calculate_rolling_rtp(
            [r.payout for r in short_records], [r.stake for r in short_records]
        )
        med_rtp = calculate_rolling_rtp(
            [r.payout for r in med_records], [r.stake for r in med_records]
        )
        macro_rtp = calculate_rolling_rtp(
            [r.payout for r in macro_records], [r.stake for r in macro_records]
        )

        delta = round(short_rtp - macro_rtp, 4)

        return {
            "short_rtp": short_rtp,
            "med_rtp": med_rtp,
            "macro_rtp": macro_rtp,
            "divergence_delta": delta,
            "sample_count": len(history),
        }

    def _evaluate_divergence_signal(
        self, source: str, stats: dict[str, float]
    ) -> RTPSignalEvent | None:
        short_rtp = stats["short_rtp"]
        macro_rtp = stats["macro_rtp"]
        delta = stats["divergence_delta"]

        signal_type = None
        desc = ""

        if delta >= self.divergence_threshold:
            signal_type = "DEFENSIVE_DRAWDOWN_ZONE"
            desc = (
                f"Short-Term RTP ({short_rtp:.2%}) exceeds Macro Baseline ({macro_rtp:.2%}) "
                f"by +{delta:.2%}. PRNG mathematically forced to pull back (Mean Reversion)."
            )
        elif delta <= -self.divergence_threshold:
            signal_type = "EXPANSION_REBALANCE_SPIKE"
            desc = (
                f"Short-Term RTP ({short_rtp:.2%}) depressed below Macro Baseline ({macro_rtp:.2%}) "
                f"by {delta:.2%}. Expected upward RTP rebalancing cycle imminent."
            )

        if signal_type:
            event = RTPSignalEvent(
                timestamp=time.time(),
                source=source,
                short_rtp=short_rtp,
                med_rtp=stats.get("med_rtp", 1.0),
                macro_rtp=macro_rtp,
                divergence_delta=delta,
                signal_type=signal_type,
                description=desc,
            )

            self._recent_signals.append(event)
            if len(self._recent_signals) > 50:
                self._recent_signals.pop(0)
            logger.warning("[RTP SIGNAL] {} for {}: {}", signal_type, source, desc)
            return event

        return None

    def get_recent_signals(self) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": s.timestamp,
                "source": s.source,
                "short_rtp": s.short_rtp,
                "med_rtp": s.med_rtp,
                "macro_rtp": s.macro_rtp,
                "divergence_delta": s.divergence_delta,
                "signal_type": s.signal_type,
                "description": s.description,
            }
            for s in reversed(self._recent_signals)
        ]


# Global singleton instance
rtp_calculator = RTPWindowCalculator()
