"""
test_ingestion_module.py
------------------------
Unit test suite for High-Speed Swarm Ingestion Module (ingestor_schema.py & ingestion_client.py).
"""

import pytest
from ingestor_schema import TelemetryPacket
from ingestion_client import SwarmIngestionClient, ingestion_client


def test_telemetry_packet_pydantic_validation():
    packet = TelemetryPacket(
        bot_id="w01",
        league_id="betking_v_league",
        theoretical_odds=2.10,
        actual_payout=210.0,
        is_win=True,
        raw_outcome_data={"event_name": "Man City vs Chelsea", "cycle": 5},
    )
    assert packet.bot_id == "w01"
    assert packet.league_id == "betking_v_league"
    assert packet.theoretical_odds == 2.10
    assert packet.actual_payout == 210.0
    assert packet.is_win is True

    supabase_dict = packet.to_supabase_dict()
    assert supabase_dict["source"] == "betking_v_league"
    assert supabase_dict["proxy_label"] == "w01"
    assert supabase_dict["actual_payout"] == 210.0


@pytest.mark.asyncio
async def test_swarm_ingestion_client_batching_and_flush():
    client = SwarmIngestionClient(batch_size=5, flush_interval_secs=10.0)

    # Ingest 4 matches (should stay in buffer)
    for i in range(4):
        ok = await client.ingest_match(
            bot_id="w02",
            league_id="betano_v_league",
            theoretical_odds=1.95,
            actual_payout=195.0,
            is_win=True,
            raw_outcome_data={"event_name": f"Event_{i}", "cycle": i},
        )
        assert ok is True

    assert client.buffered_count == 4

    # Flush batch manually
    flushed = await client.flush_batch()
    assert flushed == 4
    assert client.buffered_count == 0


@pytest.mark.asyncio
async def test_swarm_ingestion_client_retry_queue():
    client = SwarmIngestionClient(batch_size=2, flush_interval_secs=10.0)

    await client.ingest_match(
        bot_id="w03",
        league_id="msport_v_league",
        theoretical_odds=2.50,
        actual_payout=0.0,
        is_win=False,
        raw_outcome_data={"event_name": "Test Match"},
    )
    assert client.buffered_count == 1
