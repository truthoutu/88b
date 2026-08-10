"""
ingestor_schema.py
------------------
Strict Pydantic schema definition for High-Speed Swarm Telemetry Ingestion Packets.
Ensures perfectly uniform data payloads across all 9 background observer bots
before streaming into the Supabase Vault and rtp_engine.py.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class TelemetryPacket(BaseModel):
    """
    Strict schema for scraped telemetry match outcome packets.
    """
    bot_id: str = Field(..., description="Worker bot identifier (e.g. w00, w01)")
    league_id: str = Field(..., description="Virtual league / target platform identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="Precise UTC timestamp of match completion",
    )
    theoretical_odds: float = Field(default=1.95, description="Initial theoretical odds offered")
    actual_payout: float = Field(default=0.0, description="Actual amount paid out upon win")
    is_win: bool = Field(default=False, description="Boolean flag indicating winning state")
    raw_outcome_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON blob for secondary entropy & pattern analysis",
    )

    @field_validator("theoretical_odds")
    @classmethod
    def validate_odds(cls, v: float) -> float:
        if v <= 1.0:
            return 1.95
        return round(v, 4)

    @field_validator("actual_payout")
    @classmethod
    def validate_payout(cls, v: float) -> float:
        return round(max(0.0, v), 2)

    def to_supabase_dict(self) -> Dict[str, Any]:
        """Convert packet to Supabase table row format."""
        return {
            "timestamp": self.timestamp,
            "source": self.league_id,
            "event_name": self.raw_outcome_data.get("event_name", f"Match_{self.bot_id}"),
            "numeric_result": str(self.theoretical_odds),
            "cycle": int(self.raw_outcome_data.get("cycle", 0)),
            "proxy_label": self.bot_id,
            "synthetic_stake": float(self.raw_outcome_data.get("synthetic_stake", 100.0)),
            "actual_payout": self.actual_payout,
            "house_margin": round(max(0.02, 1.0 - (1.0 / self.theoretical_odds)), 4),
        }
