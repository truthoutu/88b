# Real-Time Dashboard Scraper

A **high-throughput, Playwright-based Python scraper** for live web dashboards emitting real-time updating HTML tables. Designed for longitudinal time-series data collection with stateful proxy rotation and network lightening.

## Architecture & Enterprise Features (M1 + M2 + M3)

### M1: Shared Browser & Concurrency Pool
- **Single Chromium Process**: Avoids process-per-worker overhead by running a single shared Chromium instance across all worker threads.
- **Context Isolation**: Each worker operates within an independent `BrowserContext` with randomized browser fingerprints (User-Agent, viewport size, locale, timezone, device scale factor) and anti-detection scripts (`navigator.webdriver` spoofing).
- **Concurrency Semaphore**: Workers are bounded by `--workers N` via an `asyncio.Semaphore` context pool.

### M2: Stateful Proxy Health & Circuit Breakers (`ProxyManager`)
- **Proxy Health States**: `HEALTHY`, `IN_USE`, `COOLING`, `QUARANTINED`.
- **Error Taxonomy**: Automatically classifies network failures (`AUTH`, `RATE_LIMIT`, `PROXY_HANDSHAKE`, `NETWORK_LATENCY`, `PACKET_LOSS`).
- **Circuit Breakers**: Immediately quarantines `AUTH` and `PROXY_HANDSHAKE` failures; trips breaker after 3 consecutive failures.
- **Quarantine Auto-Recovery**: Quarantined proxies auto-recover to `HEALTHY` when quarantine window expires.
- **Context Recycling**: Network errors trigger proxy release, context tear-down, and automatic rental of fresh proxies from the pool.

### M3: Network Lightening (Resource Interception)
- **Bandwidth Optimization**: Intercepts requests via `context.route("**/*")` to abort non-essential assets (`image`, `font`, `media`, `stylesheet`, `imageset`, `beacon`, `csp_report`).
- **Essential Asset Passthrough**: Keeps essential data transfer channels open (`document`, `xhr`, `fetch`, `script`) to maximize Webshare proxy throughput.

---

## Quick start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
playwright install chromium
```

### 2. Run unit test suite

```powershell
pytest
```

### 3. Production Scraper & Monitor Execution

```powershell
# Monitor mode: auto-loads targets.json & proxies.txt
python scraper.py monitor

# Explicit target file, proxy file, worker count, and save cycle
python scraper.py --target-file targets.json --proxy-file proxies.txt --workers 4 --interval 60
```

---

## Scraper CLI reference

```text
usage: scraper.py [{run,monitor}] [-h] [--url URL] [--table-id TABLE_ID]
                  [--targets SPEC] [--target-file PATH] [--output-dir PATH]
                  [--interval SECS] [--max-cycles N] [--show-browser]
                  [--log-level LEVEL] [--proxy-file PATH] [--workers N]
                  [--worker-timeout SECS]

positional arguments:
  {run,monitor}    Command mode: 'run' (default) or 'monitor' (auto-loads targets.json and proxies.txt)

options:
  --url            Dashboard URL (default: http://localhost:8765)
  --table-id       HTML <table id="…"> (default: metrics-table)
  --targets        Target definition in name:url[:table_id] format (repeatable)
  --target-file    JSON file with array of {name, url, table_id?} objects
  --output-dir     CSV output directory (default: ./scraped_data)
  --interval       Save cycle in seconds (default: 60)
  --max-cycles     Stop after N cycles; 0 = run forever (default: 0)
  --show-browser   Run Chromium in non-headless mode
  --log-level      Loguru verbosity (default: INFO)
  --proxy-file     Text file with one proxy per line (http://user:pass@host:port)
  --workers        Max concurrent browser context pool slots (default: 0 = 1 per target x proxy)
  --worker-timeout Per-worker page navigation timeout in seconds (default: 30)
```


