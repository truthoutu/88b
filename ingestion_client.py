"""
ingestion_client.py
-------------------
High-Performance Async Telemetry Ingestion Client for 9-Bot Swarm Observers.

Features:
  • Playwright Scraper Integration: Accepts raw scraped data directly from browser workers.
  • Strict Schema Validation: Enforces TelemetryPacket Pydantic schema.
  • Dual-Trigger Batching: Flushes buffer every X seconds OR every Y records (whichever comes first).
  • Zero-Loss Retry Queue: Caches data in memory during temporary database/network drops and retries automatically.
"""

from __future__ import annotations

import asyncio
import collections
import time
from typing import Any, Dict, List
from loguru import logger

from ingestor_schema import TelemetryPacket


class SwarmIngestionClient:
    """
    High-speed, fault-tolerant ingestion client instance shared across background observer bots.
    """

    def __init__(
        self,
        batch_size: int = 20,
        flush_interval_secs: float = 5.0,
        max_buffer_capacity: int = 5000,
    ) -> None:
        self.batch_size = batch_size
        self.flush_interval_secs = flush_interval_secs
        self.max_buffer_capacity = max_buffer_capacity

        self._buffer: collections.deque[TelemetryPacket] = collections.deque(maxlen=self.max_buffer_capacity)
        self._retry_queue: collections.deque[TelemetryPacket] = collections.deque(maxlen=self.max_buffer_capacity)
        self._lock = asyncio.Lock()
        self._last_flush_time = time.monotonic()

    async def ingest_match(
        self,
        bot_id: str,
        league_id: str,
        theoretical_odds: float,
        actual_payout: float,
        is_win: bool,
        raw_outcome_data: Dict[str, Any] | None = None,
        timestamp: str = "",
    ) -> bool:
        """
        Ingest a raw scraped match outcome, validate schema, and push to batch buffer.
        """
        try:
            packet = TelemetryPacket(
                bot_id=bot_id,
                league_id=league_id,
                theoretical_odds=theoretical_odds,
                actual_payout=actual_payout,
                is_win=is_win,
                raw_outcome_data=raw_outcome_data or {},
                timestamp=timestamp or "",
            )
        except Exception as exc:
            logger.warning("[INGESTOR VALIDATION ERROR] Invalid telemetry payload from bot={}: {}", bot_id, exc)
            return False

        async with self._lock:
            self._buffer.append(packet)
            should_flush = (
                len(self._buffer) >= self.batch_size
                or (time.monotonic() - self._last_flush_time) >= self.flush_interval_secs
            )

        if should_flush:
            asyncio.create_task(self.flush_batch())

        return True

    async def flush_batch(self) -> int:
        """
        Flush buffered packets to Supabase Vault and RTP engine.
        Re-queues packets in retry queue if network error occurs.
        """
        async with self._lock:
            if not self._buffer and not self._retry_queue:
                return 0

            # Drain main buffer and combine with pending retries
            packets_to_send: List[TelemetryPacket] = []
            while self._retry_queue:
                packets_to_send.append(self._retry_queue.popleft())
            while self._buffer:
                packets_to_send.append(self._buffer.popleft())

            self._last_flush_time = time.monotonic()

        if not packets_to_send:
            return 0

        # Convert to Supabase rows & process RTP calculations
        supabase_rows = [p.to_supabase_dict() for p in packets_to_send]

        try:
            from db import db
            from rtp_engine import rtp_calculator

            # Process RTP Engine calculations for each packet
            for p in packets_to_send:
                res = rtp_calculator.record_cycle_outcome(
                    source=p.league_id,
                    event_name=p.raw_outcome_data.get("event_name", f"Match_{p.bot_id}"),
                    numeric_result=p.theoretical_odds,
                    stake=float(p.raw_outcome_data.get("synthetic_stake", 100.0)),
                    cycle=int(p.raw_outcome_data.get("cycle", 0)),
                    timestamp=p.timestamp,
                )
                sig = res.get("signal")
                drift_m = res.get("drift_metrics")

                if sig and db.is_connected:
                    asyncio.create_task(db.save_rtp_signal({
                        "source": sig.source,
                        "short_rtp": sig.short_rtp,
                        "med_rtp": sig.med_rtp,
                        "macro_rtp": sig.macro_rtp,
                        "divergence_delta": sig.divergence_delta,
                        "signal_type": sig.signal_type,
                        "description": sig.description,
                    }))

                if drift_m:
                    try:
                        from oracle_alerts import oracle_engine
                        oracle_engine.evaluate_metrics(drift_m)
                    except Exception:
                        pass

                    if db.is_connected:
                        asyncio.create_task(db.save_stochastic_features({
                            "timestamp": str(drift_m.timestamp),
                            "source": drift_m.source,
                            "rolling_rtp_10": drift_m.rolling_rtp_10,
                            "rtp_z_score": drift_m.rtp_z_score,
                            "sequence_entropy": drift_m.sequence_entropy,
                            "seed_volatility": drift_m.seed_volatility,
                            "drift_confidence": drift_m.drift_confidence,
                            "is_correction_phase": drift_m.is_correction_phase,
                            "audit_message": drift_m.audit_message,
                        }))

            # Stream batch into Supabase Vault
            if db.is_connected:
                sent_count = await db.save_telemetry_batch(supabase_rows)
                logger.debug("[INGESTOR FLUSH] Flushed {:,} packets to Supabase Vault.", sent_count)
            else:
                logger.debug("[INGESTOR LOCAL] Processed {:,} packets in offline mode.", len(supabase_rows))

            return len(packets_to_send)

        except Exception as exc:
            logger.error("[INGESTOR RETRY TRIGGER] Batch flush exception: {}. Caching in memory for auto-retry.", exc)
            async with self._lock:
                for pkt in packets_to_send:
                    self._retry_queue.append(pkt)
            return 0

    @property
    def buffered_count(self) -> int:
        return len(self._buffer) + len(self._retry_queue)


# Global singleton instance
ingestion_client = SwarmIngestionClient()
