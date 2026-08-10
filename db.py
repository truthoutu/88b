"""
db.py
----
PostgreSQL / Supabase async database manager for telemetry persistence.
Integrates official `supabase-py` client with automatic `.env` credential loading.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

try:
    from supabase import create_client, Client
    HAVE_SUPABASE = True
except ImportError:
    HAVE_SUPABASE = False

try:
    import asyncpg
    HAVE_ASYNCPG = True
except ImportError:
    HAVE_ASYNCPG = False


class TelemetryDatabase:
    """Supabase / PostgreSQL connection pool manager with fallback."""

    def __init__(self) -> None:
        self._pool: Any | None = None
        self._supabase_client: Client | None = None
        self._db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
        self._supabase_url = os.getenv("SUPABASE_URL")
        self._supabase_key = os.getenv("SUPABASE_KEY")
        self._is_connected = False

    async def connect(self) -> bool:
        """Establish Supabase client or asyncpg connection pool."""
        # 1. Official Supabase-py Client
        if HAVE_SUPABASE and self._supabase_url and self._supabase_key:
            try:
                self._supabase_client = create_client(self._supabase_url, self._supabase_key)
                self._is_connected = True
                logger.success("Connected to Supabase client: {}", self._supabase_url)
                return True
            except Exception as exc:
                logger.warning("Supabase client initialization variance: {}", exc)

        # 2. Direct PostgreSQL / asyncpg connection pool fallback
        if HAVE_ASYNCPG and self._db_url:
            try:
                self._pool = await asyncpg.create_pool(
                    dsn=self._db_url,
                    min_size=2,
                    max_size=10,
                    command_timeout=10,
                )
                self._is_connected = True
                logger.success("Connected to PostgreSQL database pool.")
                await self._init_tables()
                return True
            except Exception as exc:
                logger.warning("Database pool connection failed ({}). Continuing in local fallback mode.", exc)

        logger.info("Running DB layer in offline fallback mode (local CSV persistence active).")
        self._is_connected = False
        return False

    async def _init_tables(self) -> None:
        """Execute DDL statements if using raw PostgreSQL connection."""
        if not self._pool:
            return
        schema_sql = """
        CREATE TABLE IF NOT EXISTS telemetry_scrapes (
            id BIGSERIAL PRIMARY KEY,
            timestamp VARCHAR(100) NOT NULL,
            source VARCHAR(100) NOT NULL,
            event_name VARCHAR(255) NOT NULL,
            numeric_result VARCHAR(100),
            cycle INT NOT NULL,
            proxy_label VARCHAR(100) DEFAULT 'direct',
            synthetic_stake NUMERIC(10,2) DEFAULT 100.00,
            actual_payout NUMERIC(10,2) DEFAULT 0.00,
            house_margin NUMERIC(6,4) DEFAULT 0.05,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(schema_sql)
        except Exception as exc:
            logger.warning("Could not verify database table schema: {}", exc)

    async def save_telemetry_batch(
        self,
        rows: list[dict[str, Any]],
    ) -> int:
        """Stream-insert a batch of telemetry rows into Supabase/PostgreSQL."""
        if not self._is_connected or not rows:
            return 0

        # A. Official Supabase-py API
        if self._supabase_client:
            try:
                self._supabase_client.table("telemetry_scrapes").insert(rows).execute()
                logger.debug("Persisted {:,} telemetry rows to Supabase table.", len(rows))
                return len(rows)
            except Exception as exc:
                logger.error("Failed to insert telemetry batch into Supabase: {}", exc)
                return 0

        # B. Raw PostgreSQL Pool
        if self._pool:
            insert_sql = """
            INSERT INTO telemetry_scrapes (timestamp, source, event_name, numeric_result, cycle, proxy_label, synthetic_stake, actual_payout, house_margin)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            tuples = [
                (
                    r.get("timestamp", ""),
                    r.get("source", "unknown"),
                    r.get("event_name", ""),
                    str(r.get("numeric_result", "")),
                    int(r.get("cycle", 0)),
                    r.get("proxy_label", "direct"),
                    float(r.get("synthetic_stake", 100.0)),
                    float(r.get("actual_payout", 0.0)),
                    float(r.get("house_margin", 0.05)),
                )
                for r in rows
            ]
            try:
                async with self._pool.acquire() as conn:
                    await conn.executemany(insert_sql, tuples)
                logger.debug("Persisted {:,} telemetry rows to PostgreSQL pool.", len(tuples))
                return len(tuples)
            except Exception as exc:
                logger.error("Failed to insert telemetry batch into PostgreSQL pool: {}", exc)
                return 0

        return 0

    async def save_rtp_signal(self, signal_dict: dict[str, Any]) -> bool:
        """Persist a mean reversion RTP divergence signal event into Supabase."""
        if not self._is_connected:
            return False

        if self._supabase_client:
            try:
                self._supabase_client.table("rtp_divergence_signals").insert([signal_dict]).execute()
                logger.info("Saved RTP divergence signal to Supabase Vault.")
                return True
            except Exception as exc:
                logger.error("Failed to save RTP signal to Supabase: {}", exc)
                return False

        if self._pool:
            insert_sql = """
            INSERT INTO rtp_divergence_signals (source, short_rtp, med_rtp, macro_rtp, divergence_delta, signal_type, description)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        insert_sql,
                        signal_dict.get("source", "unknown"),
                        float(signal_dict.get("short_rtp", 1.0)),
                        float(signal_dict.get("med_rtp", 1.0)),
                        float(signal_dict.get("macro_rtp", 1.0)),
                        float(signal_dict.get("divergence_delta", 0.0)),
                        signal_dict.get("signal_type", "UNKNOWN"),
                        signal_dict.get("description", ""),
                    )
                logger.info("Saved RTP divergence signal to PostgreSQL pool.")
                return True
            except Exception as exc:
                logger.error("Failed to save RTP signal to PostgreSQL pool: {}", exc)
                return False

        return False

    async def save_stochastic_features(self, metrics_dict: dict[str, Any]) -> bool:
        """Persist aggregated stochastic drift features directly into Supabase vault."""
        if not self._is_connected:
            return False

        if self._supabase_client:
            try:
                self._supabase_client.table("stochastic_drift_features").insert([metrics_dict]).execute()
                logger.debug("Saved stochastic drift features to Supabase.")
                return True
            except Exception as exc:
                logger.error("Failed to save stochastic drift features to Supabase: {}", exc)
                return False

        if self._pool:
            insert_sql = """
            INSERT INTO stochastic_drift_features (timestamp, source, rolling_rtp_10, rtp_z_score, sequence_entropy, seed_volatility, drift_confidence, is_correction_phase, audit_message)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        insert_sql,
                        str(metrics_dict.get("timestamp", time.time())),
                        metrics_dict.get("source", "unknown"),
                        float(metrics_dict.get("rolling_rtp_10", 1.0)),
                        float(metrics_dict.get("rtp_z_score", 0.0)),
                        float(metrics_dict.get("sequence_entropy", 1.0)),
                        float(metrics_dict.get("seed_volatility", 1.0)),
                        float(metrics_dict.get("drift_confidence", 0.0)),
                        bool(metrics_dict.get("is_correction_phase", False)),
                        metrics_dict.get("audit_message", ""),
                    )
                logger.debug("Saved stochastic drift features to PostgreSQL pool.")
                return True
            except Exception as exc:
                logger.error("Failed to save stochastic drift features to PostgreSQL pool: {}", exc)
                return False

        return False

    async def get_aggregated_drift_features(self, limit: int = 50) -> list[dict]:
        """Query aggregated time-series drift features directly from Supabase vault."""
        if not self._is_connected:
            return []

        if self._supabase_client:
            try:
                res = self._supabase_client.table("stochastic_drift_features").select("*").order("id", desc=True).limit(limit).execute()
                return res.data or []
            except Exception as exc:
                logger.error("Failed to query stochastic features from Supabase: {}", exc)
                return []

        if self._pool:
            query = """
            SELECT timestamp, source, rolling_rtp_10, rtp_z_score, sequence_entropy, seed_volatility, drift_confidence, is_correction_phase, audit_message, created_at
            FROM stochastic_drift_features
            ORDER BY id DESC
            LIMIT $1
            """
            try:
                async with self._pool.acquire() as conn:
                    records = await conn.fetch(query, limit)
                    return [dict(rec) for rec in records]
            except Exception as exc:
                logger.error("Failed to query stochastic features from PostgreSQL: {}", exc)
                return []

        return []

    async def get_recent_scrapes(self, limit: int = 50) -> list[dict]:

        """Fetch the most recent scraped records from Supabase."""
        if not self._is_connected:
            return []

        if self._supabase_client:
            try:
                res = self._supabase_client.table("telemetry_scrapes").select("*").order("id", desc=True).limit(limit).execute()
                return res.data or []
            except Exception as exc:
                logger.error("Failed to query telemetry records from Supabase: {}", exc)
                return []

        if self._pool:
            query = """
            SELECT timestamp, source, event_name, numeric_result, cycle, proxy_label, created_at
            FROM telemetry_scrapes
            ORDER BY id DESC
            LIMIT $1
            """
            try:
                async with self._pool.acquire() as conn:
                    records = await conn.fetch(query, limit)
                    return [dict(rec) for rec in records]
            except Exception as exc:
                logger.error("Failed to query telemetry records from PostgreSQL pool: {}", exc)
                return []

        return []

    async def disconnect(self) -> None:
        """Close database connections."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._supabase_client = None
        self._is_connected = False
        logger.info("Database connection closed.")

    @property
    def is_connected(self) -> bool:
        return self._is_connected


# Global singleton instance
db = TelemetryDatabase()
