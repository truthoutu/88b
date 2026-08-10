-- =============================================================================
-- schema.sql
-- ----------
-- Supabase / PostgreSQL Database Schema for Scraper Telemetry Vault
-- =============================================================================

-- Target platform registry
CREATE TABLE IF NOT EXISTS targets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    url TEXT NOT NULL,
    table_id VARCHAR(100) DEFAULT 'table',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Time-series telemetry scrapes with granular economic state fields
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

-- Index for high-speed time-series analytics and de-duplication
CREATE INDEX IF NOT EXISTS idx_telemetry_source_timestamp 
    ON telemetry_scrapes (source, timestamp, event_name);

CREATE INDEX IF NOT EXISTS idx_telemetry_created_at 
    ON telemetry_scrapes (created_at DESC);

-- RTP Mean Reversion & Divergence Signal Log
CREATE TABLE IF NOT EXISTS rtp_divergence_signals (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(100) NOT NULL,
    short_rtp NUMERIC(6,4) NOT NULL,
    med_rtp NUMERIC(6,4) NOT NULL,
    macro_rtp NUMERIC(6,4) NOT NULL,
    divergence_delta NUMERIC(6,4) NOT NULL,
    signal_type VARCHAR(100) NOT NULL,
    description TEXT,
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Stochastic Drift & PRNG Auditor Aggregated Features
CREATE TABLE IF NOT EXISTS stochastic_drift_features (
    id BIGSERIAL PRIMARY KEY,
    timestamp VARCHAR(100) NOT NULL,
    source VARCHAR(100) NOT NULL,
    rolling_rtp_10 NUMERIC(8,4) NOT NULL,
    rtp_z_score NUMERIC(8,4) NOT NULL,
    sequence_entropy NUMERIC(6,4) NOT NULL,
    seed_volatility NUMERIC(8,4) NOT NULL,
    drift_confidence NUMERIC(6,4) NOT NULL,
    is_correction_phase BOOLEAN DEFAULT FALSE,
    audit_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_drift_source_created 
    ON stochastic_drift_features (source, created_at DESC);


-- Proxy health tracking log

CREATE TABLE IF NOT EXISTS proxy_health_logs (
    id BIGSERIAL PRIMARY KEY,
    proxy_label VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    fail_count INT DEFAULT 0,
    ok_count INT DEFAULT 0,
    requests INT DEFAULT 0,
    last_error TEXT,
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- System job execution history
CREATE TABLE IF NOT EXISTS system_jobs (
    job_id VARCHAR(100) PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    stopped_at TIMESTAMPTZ
);
