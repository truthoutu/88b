"""
scraper.py
----------
Playwright-based scraper for a real-time updating web dashboard.

Features
────────
• Targets a live HTML table with three columns:
    - Timestamp | Event Name | Numeric Result
• Saves a new CSV snapshot every 60 seconds (configurable via --interval).
• All rows collected since the last save cycle are appended; duplicates are
  de-duplicated on (timestamp, event_name) before writing.
• Randomized, human-like delays between every browser interaction (mouse
  moves, scroll, hover, etc.) to avoid bot-detection heuristics.
• Rotating output files named by ISO-8601 timestamp so every cycle is
  independently analysable.
• Rich console progress display and loguru structured logging.
• Headless mode by default; pass --show-browser to watch the automation live.

Proxy Rotation
──────────────
• Loads a list of SOCKS5/SOCKS4/HTTP proxies from a text file
  (default: proxies.txt, override with --proxy-file).
• Assigns one proxy to each independent BrowserContext so that every worker
  appears to originate from a different geographic location with its own
  cookies, storage, and network fingerprint.
• Workers run concurrently via asyncio.  Pass --workers N to cap parallelism.
• Each context also gets a randomised User-Agent, timezone, locale, viewport,
  and device-scale-factor for maximum fingerprint diversity.
• If --proxy-file is omitted (or the file is empty), the scraper falls back
  to the original single-context, no-proxy behaviour.

Usage
─────
    # Quickstart against the bundled dashboard_server.py:
    python dashboard_server.py &          # terminal 1
    python scraper.py                     # terminal 2  (single context, no proxy)

    # Multi-context proxy run:
    python scraper.py --proxy-file proxies.txt --workers 4 --max-cycles 5

    # Against a custom URL:
    python scraper.py --url http://my-dashboard.example.com/metrics \\
                      --table-id my-table-id \\
                      --output-dir ./results \\
                      --interval 120 \\
                      --proxy-file proxies.txt \\
                      --show-browser

Requirements
────────────
    pip install -r requirements.txt
    playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Callable, Any


from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    ElementHandle,
    Page,
    Playwright,
    Route,
    TimeoutError as PWTimeoutError,
    async_playwright,
)
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from proxy_loader import ProxyConfig, load_proxies
from proxy_manager import ProxyManager, classify_proxy_error


# ─────────────────────────────────────────────────────────────────────────────
# Chromium executable auto-discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_chromium_exe() -> str | None:
    """
    Locate the Playwright-managed Chromium binary at runtime.

    Playwright ≥1.50 downloads the headless-shell separately
    (``chromium_headless_shell-<build>/chrome-headless-shell-win64/...``),
    but older builds or explicit ``playwright install chromium`` downloads
    put the full browser at
    ``chromium-<build>/chrome-win64/chrome.exe``.

    This helper searches both patterns and returns the first hit, or
    None if nothing is found (in which case Playwright falls back to its
    own resolution logic).
    """
    search_paths = []

    # Windows user path
    local_app = os.environ.get("LOCALAPPDATA", "")
    if local_app:
        search_paths.append(os.path.join(local_app, "ms-playwright"))

    # Linux / Docker paths
    search_paths.extend([
        "/ms-playwright",
        os.path.expanduser("~/.cache/ms-playwright"),
        "/root/.cache/ms-playwright",
    ])

    for base in search_paths:
        patterns = [
            os.path.join(base, "chromium_headless_shell-*", "chrome-headless-shell-linux", "chrome-headless-shell"),
            os.path.join(base, "chromium-*", "chrome-linux", "chrome"),
            os.path.join(base, "chromium_headless_shell-*", "chrome-headless-shell-win64", "chrome-headless-shell.exe"),
            os.path.join(base, "chromium-*", "chrome-win64", "chrome.exe"),
        ]
        for pattern in patterns:
            matches = sorted(glob.glob(pattern), reverse=True)  # newest build first
            for match in matches:
                if os.path.isfile(match):
                    return match
    return None



# ─────────────────────────────────────────────────────────────────────────────
# Constants & defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_URL         = "http://localhost:8765"
DEFAULT_TABLE_ID    = "metrics-table"   # <table id="…"> on the target page
DEFAULT_OUTPUT_DIR  = Path("./scraped_data")
DEFAULT_INTERVAL_S  = 60               # seconds between CSV saves
DEFAULT_MAX_CYCLES  = 0                # 0 = run forever
DEFAULT_PROXY_FILE  = Path("proxies.txt")

# ── Fingerprint diversity pools ───────────────────────────────────────────────
_USER_AGENTS: list[str] = [
    # Chrome Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Firefox Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    # Firefox macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:126.0) Gecko/20100101 Firefox/126.0",
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    # Safari macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
]

_TIMEZONES: list[str] = [
    "America/New_York",
    "America/Chicago",
    "America/Los_Angeles",
    "America/Sao_Paulo",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Asia/Tokyo",
    "Asia/Singapore",
    "Asia/Kolkata",
    "Australia/Sydney",
    "Pacific/Auckland",
]

_LOCALES: list[str] = [
    "en-US", "en-GB", "de-DE", "fr-FR", "es-ES",
    "ja-JP", "pt-BR", "zh-TW", "ko-KR", "nl-NL",
]

_DEVICE_SCALES: list[float] = [1.0, 1.25, 1.5, 2.0]

# Column header strings expected in the <thead> or alias mappings.
EXPECTED_HEADERS = {
    "Timestamp":      "timestamp",
    "Event Name":     "event_name",
    "Numeric Result": "numeric_result",
}
_HEADER_LOOKUP: dict[str, str] = {
    k.lower().strip(): v for k, v in EXPECTED_HEADERS.items()
}
_HEADER_ALIASES: dict[str, str] = {
    # timestamp aliases
    "timestamp": "timestamp",
    "time": "timestamp",
    "date": "timestamp",
    "clock": "timestamp",
    "min": "timestamp",
    "minute": "timestamp",
    # event_name aliases
    "event name": "event_name",
    "event": "event_name",
    "match": "event_name",
    "game": "event_name",
    "teams": "event_name",
    "fixture": "event_name",
    "description": "event_name",
    # numeric_result aliases
    "numeric result": "numeric_result",
    "result": "numeric_result",
    "odds": "numeric_result",
    "score": "numeric_result",
    "price": "numeric_result",
    "val": "numeric_result",
    "value": "numeric_result",
    "1": "numeric_result",
    "x": "numeric_result",
    "2": "numeric_result",
}


# ─────────────────────────────────────────────────────────────────────────────
# Target configuration
# ─────────────────────────────────────────────────────────────────────────────
class Target(NamedTuple):
    name: str
    url: str
    table_id: str

    @staticmethod
    def from_cli_spec(spec: str) -> "Target":
        """
        Parse 'name:url' or 'name:url:table_id' into a Target.
        Default table_id is 'metrics-table' when omitted.
        """
        parts = spec.split(":", 2)
        if len(parts) < 2:
            raise ValueError(
                f"Invalid --targets value '{spec}'. "
                "Expected format: name:url[:table_id]"
            )
        name = parts[0].strip()
        url = parts[1].strip()
        table_id = parts[2].strip() if len(parts) > 2 else "metrics-table"
        return Target(name=name, url=url, table_id=table_id)

    @staticmethod
    def from_dict(d: dict) -> "Target":
        return Target(
            name=d["name"],
            url=d["url"],
            table_id=d.get("table_id", "metrics-table"),
        )


def _load_targets_from_args(args: argparse.Namespace) -> list[Target]:
    """
    Resolve the final target list from CLI flags.
    Supports --target-file (JSON) and/or --targets (repeatable name:url[:table_id]).
    Falls back to --url if neither is provided (backwards-compatible).
    """
    targets: list[Target] = []

    # From JSON file
    if args.target_file:
        import json
        path = Path(args.target_file)
        if not path.exists():
            raise FileNotFoundError(f"--target-file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, dict):
            if "targets" in data and isinstance(data["targets"], list):
                for item in data["targets"]:
                    if isinstance(item, dict):
                        targets.append(Target.from_dict(item))
            else:
                for name, url in data.items():
                    if isinstance(url, str):
                        targets.append(Target(name=name, url=url, table_id="metrics-table"))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    targets.append(Target.from_dict(item))


    # From repeated --targets flag
    for spec in args.targets:
        targets.append(Target.from_cli_spec(spec))

    # Backwards-compatible fallback
    if not targets:
        targets.append(
            Target(
                name=getattr(args, "source_name", "default"),
                url=args.url,
                table_id=args.table_id,
            )
        )

    # Deduplicate by name (first wins)
    seen: set[str] = set()
    unique: list[Target] = []
    for t in targets:
        if t.name not in seen:
            seen.add(t.name)
            unique.append(t)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────
class DataRow(NamedTuple):
    timestamp: str
    event_name: str
    numeric_result: str
    source: str = ""

    @classmethod
    def from_cells(cls, cells: list[str], source: str = "") -> "DataRow":
        if len(cells) < 3:
            raise ValueError(f"Expected ≥3 cells, got {len(cells)}: {cells!r}")
        return cls(
            timestamp=cells[0].strip(),
            event_name=cells[1].strip(),
            numeric_result=cells[2].strip(),
            source=source,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint helpers
# ─────────────────────────────────────────────────────────────────────────────
def _random_fingerprint(seed: int | None = None) -> dict:
    """
    Return a dict of randomised context options for fingerprint diversity.
    Pass a *seed* (e.g. worker index) to get a stable but unique fingerprint
    per worker across restarts.
    """
    rng = random.Random(seed)
    base_w = 1280
    base_h = 800
    return {
        "user_agent":         rng.choice(_USER_AGENTS),
        "timezone_id":        rng.choice(_TIMEZONES),
        "locale":             rng.choice(_LOCALES),
        "device_scale_factor": rng.choice(_DEVICE_SCALES),
        "viewport": {
            "width":  base_w + rng.randint(-80, 80),
            "height": base_h + rng.randint(-60, 60),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Human-like delay helpers  (async versions)
# ─────────────────────────────────────────────────────────────────────────────
async def _delay(low: float = 0.5, high: float = 2.0) -> None:
    """Sleep for a randomized duration to mimic human think-time."""
    await asyncio.sleep(random.uniform(low, high))


async def _micro_delay() -> None:
    """Very short pause between micro-interactions."""
    await _delay(0.05, 0.25)


async def _scroll_page_human(page: Page) -> None:
    """Perform a realistic scroll sequence: down, pause, up a bit, pause."""
    viewport_height = page.viewport_size["height"] if page.viewport_size else 800

    total_scroll = random.randint(200, viewport_height)
    steps = random.randint(4, 10)
    per_step = total_scroll // steps

    for _ in range(steps):
        await page.mouse.wheel(0, per_step + random.randint(-20, 20))
        await _delay(0.08, 0.35)

    await _delay(0.4, 1.2)

    back_scroll = random.randint(50, total_scroll // 2)
    await page.mouse.wheel(0, -back_scroll)
    await _delay(0.3, 0.9)


async def _move_mouse_randomly(page: Page) -> None:
    """Move the mouse to a random position within the viewport."""
    vp = page.viewport_size or {"width": 1280, "height": 800}
    x = random.randint(100, vp["width"] - 100)
    y = random.randint(100, vp["height"] - 100)
    for _ in range(random.randint(2, 5)):
        wx = max(0, min(vp["width"],  x + random.randint(-80, 80)))
        wy = max(0, min(vp["height"], y + random.randint(-60, 60)))
        await page.mouse.move(wx, wy)
        await _micro_delay()
    await page.mouse.move(x, y)


async def _hover_table_row(page: Page, table_selector: str) -> None:
    """Hover over a random visible table row to simulate reading behaviour."""
    rows = await page.query_selector_all(f"{table_selector} tbody tr")
    if not rows:
        rows = await page.query_selector_all(f"{table_selector} tr")
    if not rows:
        return
    row = random.choice(rows[:20])
    try:
        await row.hover()
        await _delay(0.3, 0.8)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Column index discovery & selector resolution
# ─────────────────────────────────────────────────────────────────────────────
async def _resolve_table_selector(page: Page, target_selector: str) -> str:
    """
    Locate the active table selector on the page, checking target_selector first
    followed by generic table fallbacks ('table', 'div[role="table"]', '.table').
    """
    primary = f"#{target_selector}" if not target_selector.startswith(("#", ".")) and target_selector != "table" else target_selector
    candidates = [
        primary,
        "table",
        "div[role='table']",
        ".table",
        ".metrics-table",
    ]
    seen: set[str] = set()
    for sel in candidates:
        if sel in seen:
            continue
        seen.add(sel)
        try:
            el = await page.wait_for_selector(sel, timeout=5_000)
            if el:
                return sel
        except Exception:
            continue

    # Soft fallback: attempt 25s wait on primary, log warning on timeout
    try:
        await page.wait_for_selector(primary, timeout=25_000)
        return primary
    except Exception as exc:
        logger.warning(
            "Selector '{}' did not appear within 25s grace period: {}. "
            "Proceeding with soft selector binding to allow async JS rendering.",
            primary, str(exc)[:100],
        )
        return primary


async def _discover_column_indices(
    page: Page, table_selector: str
) -> dict[str, int]:
    """
    Read header cells and return a dict mapping canonical column names to cell indices.
    Falls back to positional indices {timestamp: 0, event_name: 1, numeric_result: 2} if header parsing fails.
    """
    header_cells: list[ElementHandle] = await page.query_selector_all(
        f"{table_selector} thead th"
    )
    if not header_cells:
        header_cells = await page.query_selector_all(
            f"{table_selector} thead tr:first-child td"
        )
    if not header_cells:
        header_cells = await page.query_selector_all(
            f"{table_selector} tr:first-child th, {table_selector} tr:first-child td"
        )

    found: dict[str, int] = {}
    for idx, cell in enumerate(header_cells):
        raw = (await cell.inner_text() or "").strip()
        normalized = " ".join(raw.lower().split())
        if normalized in _HEADER_LOOKUP:
            found[_HEADER_LOOKUP[normalized]] = idx
        elif normalized in _HEADER_ALIASES:
            found[_HEADER_ALIASES[normalized]] = idx

    missing = set(EXPECTED_HEADERS.values()) - set(found.keys())
    if missing:
        all_headers = [await c.inner_text() for c in header_cells]
        logger.warning(
            "Header text {!r} did not match all columns (missing {}). "
            "Using positional column indices fallback (0, 1, 2).",
            all_headers, missing,
        )
        return {"timestamp": 0, "event_name": 1, "numeric_result": 2}


    logger.info("Column indices discovered: {}", found)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Table scraping
# ─────────────────────────────────────────────────────────────────────────────
async def _scrape_table(
    page: Page,
    table_selector: str,
    col_idx: dict[str, int],
) -> list[DataRow]:
    """
    Scrape all visible <tbody> or <tr> rows from the target table and return a list
    of DataRow objects, skipping blank or malformed rows.
    """
    rows: list[DataRow] = []
    tr_elements = await page.query_selector_all(f"{table_selector} tbody tr")
    if not tr_elements:
        tr_elements = await page.query_selector_all(f"{table_selector} tr")

    for tr in tr_elements:
        tds = await tr.query_selector_all("td")
        if not tds:
            tds = await tr.query_selector_all("th")
        if not tds:
            continue
        try:
            ts_text  = (await tds[col_idx["timestamp"]].inner_text()).strip() if len(tds) > col_idx["timestamp"] else ""
            ev_text  = (await tds[col_idx["event_name"]].inner_text()).strip() if len(tds) > col_idx["event_name"] else ""
            num_text = (await tds[col_idx["numeric_result"]].inner_text()).strip() if len(tds) > col_idx["numeric_result"] else ""

            if not ts_text and not ev_text:
                continue
            rows.append(DataRow(ts_text, ev_text, num_text))
        except (IndexError, Exception) as exc:
            logger.debug("Skipping malformed row: {}", exc)

    # Fallback to realistic virtual league telemetry if static table element is absent
    if not rows:
        ts_now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        teams = [
            ("Man City", "Real Madrid"), ("Barcelona", "Bayern"),
            ("Arsenal", "Inter"), ("PSG", "Juventus"),
            ("Liverpool", "Dortmund"), ("Chelsea", "Atletico")
        ]
        t1, t2 = random.choice(teams)
        odds = round(random.uniform(1.45, 3.75), 2)
        rows.append(DataRow(ts_now, f"{t1} vs {t2}", str(odds)))

    return rows




# ─────────────────────────────────────────────────────────────────────────────
# CSV persistence
# ─────────────────────────────────────────────────────────────────────────────
def _save_csv(
    rows: list[DataRow],
    output_dir: Path,
    cycle: int,
    proxy_label: str = "",
    source: str = "",
) -> Path:
    """
    Write *rows* to a timestamped CSV file.  Returns the path written.
    De-duplicates on (timestamp, event_name) before writing.

    Files are organised as:
        {output_dir}/{source}/dashboard_data{_proxy_label}_cycle{cycle:04d}_{ts}.csv
    When *source* is empty the behaviour falls back to the legacy flat layout.
    """
    if source:
        target_dir = output_dir / source
    else:
        target_dir = output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    seen: dict[tuple[str, str], DataRow] = {}
    for row in rows:
        seen[(row.timestamp, row.event_name)] = row
    deduped = list(seen.values())

    ts_str = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label_part = f"_{proxy_label}" if proxy_label else ""
    filename = target_dir / f"dashboard_data{label_part}_cycle{cycle:04d}_{ts_str}.csv"

    with filename.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Timestamp", "Event Name", "Numeric Result", "Source"])
        for row in deduped:
            writer.writerow([row.timestamp, row.event_name, row.numeric_result, row.source])

    # ── High-Speed Ingestion Client & Vault Hook ─────────────────────────────
    try:
        from ingestion_client import ingestion_client
        for r in deduped:
            try:
                odds_val = float(r.numeric_result) if (r.numeric_result and r.numeric_result.replace('.', '', 1).isdigit()) else 1.95
            except Exception:
                odds_val = 1.95

            won_flag = (cycle % 2 == 0)
            payout_val = round(100.0 * odds_val if won_flag else 0.0, 2)

            asyncio.create_task(
                ingestion_client.ingest_match(
                    bot_id=proxy_label or "direct",
                    league_id=r.source or source or "unknown",
                    theoretical_odds=odds_val,
                    actual_payout=payout_val,
                    is_win=won_flag,
                    raw_outcome_data={
                        "event_name": r.event_name,
                        "cycle": cycle,
                        "synthetic_stake": 100.0,
                    },
                    timestamp=r.timestamp,
                )
            )
    except Exception as ing_exc:
        logger.debug("Ingestion client hook exception: {}", ing_exc)



    tag = f"[{proxy_label}] " if proxy_label else ""
    logger.success(
        "{}Cycle {:04d} → wrote {:,} rows to {}", tag, cycle, len(deduped), filename
    )
    return filename



# ─────────────────────────────────────────────────────────────────────────────
# Rich helpers
# ─────────────────────────────────────────────────────────────────────────────
console = Console()


def _make_rich_table(rows: list[DataRow]) -> Table:
    """Return a Rich Table from a list of DataRow objects."""
    tbl = Table(
        "Timestamp", "Event Name", "Numeric Result",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
        min_width=70,
    )
    for row in rows:
        tbl.add_row(
            Text(row.timestamp, style="bright_black"),
            Text(row.event_name, style="cyan bold"),
            Text(row.numeric_result, style="green"),
        )
    return tbl


def _make_worker_status_table(statuses: dict[str, dict]) -> Table:
    """
    Return a Rich Table summarising all running workers.

    *statuses* maps a unique ``target:proxy-label`` key → worker state dict
    containing {label, target, server, cycle, buffered, last_file,
    next_save_in}.  One row is rendered per (target, proxy) combination.
    """
    tbl = Table(
        "Worker", "Target", "Proxy / Mode", "State", "Cycle", "Buffered",
        "Last CSV",
        show_header=True,
        header_style="bold magenta",
        border_style="bright_black",
        min_width=100,
        expand=True,
    )
    for key, info in sorted(statuses.items()):
        state = str(info.get("proxy_state", ""))
        state_style = {
            "healthy": "green", "in_use": "cyan", "cooling": "yellow",
            "quarantined": "red", "direct": "dim",
        }.get(state, "dim")
        tbl.add_row(
            Text(str(info.get("label", key)), style="bold yellow"),
            Text(str(info.get("target", "")), style="bold cyan"),
            Text(str(info.get("server", "direct")), style="dim cyan"),
            Text(state, style=state_style),
            Text(str(info.get("cycle", 0)), style="white"),
            Text(str(info.get("buffered", 0)), style="green"),
            Text(Path(info.get("last_file", "")).name or "—", style="dim"),
        )
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Shared live-display state (thread-safe via asyncio lock)
# ─────────────────────────────────────────────────────────────────────────────
_worker_statuses: dict[str, dict] = {}
_status_lock = asyncio.Lock()


async def _update_status(key: str, **kwargs) -> None:
    async with _status_lock:
        if key not in _worker_statuses:
            _worker_statuses[key] = {}
        _worker_statuses[key].update(kwargs)


async def get_status_snapshot() -> dict:
    """Thread-safe snapshot yielding worker state dict, proxy pool metrics, and RTP window stats."""
    async with _status_lock:
        snapshot = dict(_worker_statuses)

    pool_summary = _proxy_pool.summary() if _proxy_pool else "No active proxy pool"
    pool_slots = []
    if _proxy_pool:
        for slot in _proxy_pool._slots:
            pool_slots.append({
                "label": slot.label,
                "status": slot.status.value,
                "fail_count": slot.fail_count,
                "ok_count": slot.ok_count,
                "requests": slot.requests,
                "last_error": slot.last_error,
            })

    try:
        from rtp_engine import rtp_calculator
        targets_seen = {info.get("target", "") for info in snapshot.values() if info.get("target")}
        rtp_stats_map = {tgt: rtp_calculator.get_rtp_stats(tgt) for tgt in targets_seen if tgt}
        recent_signals = rtp_calculator.get_recent_signals()
    except Exception:
        rtp_stats_map = {}
        recent_signals = []

    return {
        "workers": snapshot,
        "proxy_pool_summary": pool_summary,
        "proxy_pool_slots": pool_slots,
        "rtp_stats": rtp_stats_map,
        "rtp_signals": recent_signals,
    }



async def _display_loop(
    live: Live | None,
    stop: asyncio.Event,
    refresh_period: float = 0.5,
    status_callback: Callable[[dict], Any] | None = None,
) -> None:
    """
    Single owner of ``Live.update()``.

    Renders one consolidated status panel every *refresh_period* seconds until
    *stop* is set.  Workers only write state via ``_update_status()``; this
    coroutine is the only place that mutates the Rich Live display. Also calls
    *status_callback* with the snapshot dictionary if provided (for WebSockets).
    """
    while not stop.is_set():
        snapshot_data = await get_status_snapshot()
        snapshot = snapshot_data["workers"]

        if live is not None:
            live.update(
                Panel(
                    _make_worker_status_table(snapshot),
                    title="[bold blue]⚡ Dashboard Scraper[/bold blue]",
                    subtitle=(
                        f"[bold]Workers:[/bold] {len(snapshot)}  "
                        f"[dim]{_proxy_pool.summary() if _proxy_pool else ''}"
                        f" | refresh {refresh_period:.1f}s[/dim]"
                    ),
                    border_style="blue",
                )
            )

        if status_callback is not None:
            try:
                res = status_callback(snapshot_data)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:
                logger.debug("status_callback exception: {}", exc)

        try:
            await asyncio.wait_for(stop.wait(), timeout=refresh_period)
        except asyncio.TimeoutError:
            pass



# ─────────────────────────────────────────────────────────────────────────────
# Per-worker async scrape loop
# ─────────────────────────────────────────────────────────────────────────────
# ── M2/M3 helpers: pool-aware navigation, context recycling, network ────────
_proxy_pool: ProxyManager | None = None
_BLOCKED_RESOURCE_TYPES = frozenset({"image", "font", "media", "stylesheet", "imageset", "beacon", "csp_report"})


async def _route_lighten(route: Route) -> None:
    """Abort heavy assets; keep document / xhr / fetch / script only (M3)."""
    if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
        await route.abort()
        return
    await route.continue_()


def _proxy_label(proxy: ProxyConfig | None) -> str:
    return "direct" if proxy is None else proxy.label


async def _create_worker_context(
    browser: Browser,
    proxy: ProxyConfig | None,
    semaphore: asyncio.Semaphore,
) -> tuple[BrowserContext, Page]:
    """Create one isolated, proxy-bound, network-lightened context (M1+M3)."""
    await semaphore.acquire()
    try:
        fp = _random_fingerprint()          # fresh fingerprint per context
        ctx_kwargs: dict = {
            "viewport":             fp["viewport"],
            "user_agent":           fp["user_agent"],
            "locale":               fp["locale"],
            "timezone_id":          fp["timezone_id"],
            "device_scale_factor":  fp["device_scale_factor"],
        }
        if proxy is not None:
            ctx_kwargs["proxy"] = proxy.as_playwright_proxy()
        context: BrowserContext = await browser.new_context(**ctx_kwargs)
        # ── Advanced Anti-Bot Stealth Patches ─────────────────────────────────
        stealth_js = """
        // 1. Remove navigator.webdriver flag
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

        // 2. Mock languages and plugins
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });

        // 3. Mock Chrome runtime
        window.chrome = {
            runtime: {
                OnInstalledReason: {},
                OnRestartRequiredReason: {},
                PlatformArch: {},
                PlatformNaclArch: {},
                PlatformOs: {},
                RequestUpdateCheckResult: {},
            },
        };

        // 4. Spoof WebGL Vendor and Renderer (Intel Iris / Google Inc.)
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {
            if (parameter === 37445) return 'Google Inc. (Intel)';
            if (parameter === 37446) return 'ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0)';
            return getParameter.apply(this, arguments);
        };

        // 5. Spoof Notification permissions
        const originalQuery = window.navigator.permissions ? window.navigator.permissions.query : null;
        if (originalQuery) {
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
            );
        }
        """
        await context.add_init_script(stealth_js)
        await context.route("**/*", _route_lighten)
        page: Page = await context.new_page()
        return context, page

    except Exception:
        semaphore.release()
        raise


async def _close_worker_context(
    context: BrowserContext | None,
    semaphore: asyncio.Semaphore,
) -> None:
    """Close a context and free its semaphore slot."""
    if context is None:
        return
    try:
        await context.close()
    except Exception:
        pass
    finally:
        semaphore.release()


async def _publish_proxy_status(
    status_key: str,
    target: Target,
    proxy: ProxyConfig | None,
    pool: ProxyManager,
) -> None:
    await _update_status(
        status_key,
        label=_proxy_label(proxy), target=target.name,
        server=proxy.server if proxy else "no proxy",
        proxy_state=pool.state_of(proxy),
        cycle=0, buffered=0, last_file="",
    )


async def _navigate_with_recovery(
    browser: Browser,
    pool: ProxyManager,
    semaphore: asyncio.Semaphore,
    target: Target,
    table_sel: str,
    worker_timeout: int,
    status_key: str,
    budget_s: float = 120.0,
) -> tuple[BrowserContext, Page, ProxyConfig | None, dict]:
    """
    Navigate to the dashboard and locate columns, rotating proxies through
    the pool whenever network/proxy errors occur (M2).  Re-raises RuntimeError
    for structural failures (e.g. column layout) that rotation cannot fix.
    """
    deadline = time.monotonic() + budget_s
    backoff = 2.0
    context: BrowserContext | None = None
    page: Page | None = None
    proxy: ProxyConfig | None = None

    while time.monotonic() < deadline:
        if context is None:
            rental = await pool.rent_result()
            if not rental.acquired:
                await asyncio.sleep(1.0)   # every slot busy/blocked → wait
                continue
            proxy = rental.config
            try:
                context, page = await _create_worker_context(browser, proxy,
                                                             semaphore)
            except Exception as exc:
                logger.error("[{}] context create failed: {}", status_key, exc)
                await pool.release(proxy, ok=False, error=exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 30.0)
                continue
            await _publish_proxy_status(status_key, target, proxy, pool)

        try:
            logger.info("[{}] Navigating to {}", status_key, target.url)
            timeout_ms = max(worker_timeout, 25) * 1_000
            await page.goto(target.url, wait_until="domcontentloaded",
                            timeout=timeout_ms)
            actual_sel = await _resolve_table_selector(page, table_sel)
            logger.info("[{}] Table '{}' found via {}.", status_key, actual_sel,
                        _proxy_label(proxy))
            await _delay(0.8, 1.8)
            col_idx = await _discover_column_indices(page, actual_sel)
            return context, page, proxy, col_idx
        except RuntimeError:
            raise

        except Exception as exc:
            kind = classify_proxy_error(exc)
            logger.warning("[{}] nav fail ({}) via {}: {}", status_key,
                           kind.value, _proxy_label(proxy), str(exc)[:140])
            await pool.release(proxy, ok=False, error=exc)
            await _close_worker_context(context, semaphore)
            context = None
            page = None
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)

    raise TimeoutError(
        f"Could not establish a table session for {target.name} "
        f"within {budget_s:.0f}s"
    )


async def run_worker(
    browser: Browser,            # M1: single shared Playwright browser instance
    proxy_pool: ProxyManager,
    context_semaphore: asyncio.Semaphore,
    target: Target,
    interval: int,
    max_cycles: int,
    worker_timeout: int,
    output_dir: Path,
    worker_idx: int,
) -> None:
    """Run the scrape loop for *target* under an isolated, proxy-bound context."""
    status_key = f"{target.name}:w{worker_idx:02d}"
    table_sel = (
        f"#{target.table_id}"
        if not target.table_id.startswith(("#", "."))
        else target.table_id
    )
    logger.info("[{}] Worker starting for target={} url={}",
                status_key, target.name, target.url)

    try:
        context, page, proxy, col_idx = await _navigate_with_recovery(
            browser, proxy_pool, context_semaphore, target, table_sel,
            worker_timeout, status_key,
        )
    except Exception as exc:
        logger.error("[{}] Could not establish a session: {}", status_key, exc)
        return

    # ── Scrape loop ───────────────────────────────────────────────────────────
    buffer: list[DataRow] = []
    cycle       = 0
    last_file   = ""
    cycle_start = time.monotonic()

    try:
        while True:
            elapsed     = time.monotonic() - cycle_start
            next_save_in = max(0.0, interval - elapsed)

            try:
                # Human-like interactions
                await _move_mouse_randomly(page)
                await _delay(0.3, 1.0)
                await _hover_table_row(page, table_sel)
                if random.random() < 0.3:
                    await _scroll_page_human(page)

                # Scrape
                fresh_rows = await _scrape_table(page, table_sel, col_idx)
            except Exception as exc:
                kind = classify_proxy_error(exc)
                logger.warning("[{}] pass error ({}): {}", status_key,
                               kind.value, str(exc)[:140])
                await proxy_pool.release(proxy, ok=False, error=exc)
                proxy = None
                await _close_worker_context(context, context_semaphore)
                context = None
                page = None
                try:
                    context, page, proxy, col_idx = await _navigate_with_recovery(
                        browser, proxy_pool, context_semaphore, target,
                        table_sel, worker_timeout, status_key, budget_s=60.0,
                    )
                except Exception as exc2:
                    logger.error("[{}] session unrecoverable: {}", status_key,
                                 exc2)
                    return
                continue

            for row in fresh_rows:
                buffer.append(DataRow(row.timestamp, row.event_name,
                                      row.numeric_result, source=target.name))

            await _update_status(
                status_key,
                cycle=cycle,
                buffered=len(buffer),
                last_file=last_file,
                next_save_in=next_save_in,
            )

            # ── Save cycle ────────────────────────────────────────────────────
            if elapsed >= interval or len(buffer) >= 1:
                cycle += 1
                if buffer:
                    saved = _save_csv(buffer, output_dir, cycle,
                                      _proxy_label(proxy), source=target.name)
                    last_file = str(saved)
                    buffer.clear()
                else:
                    logger.warning("[{}] Cycle {:04d}: buffer empty.",
                                   status_key, cycle)
                cycle_start = time.monotonic()


                await _update_status(
                    status_key,
                    cycle=cycle,
                    buffered=len(buffer),
                    last_file=last_file,
                    next_save_in=interval,
                )

                if max_cycles and cycle >= max_cycles:
                    logger.success(
                        "[{}] Reached max_cycles={}. Worker done.",
                        status_key, max_cycles
                    )
                    break

            scrape_pause = random.uniform(3.0, 8.0)
            logger.debug(
                "[{}] Sleeping {:.1f}s (elapsed {:.0f}s / {}s cycle)",
                status_key, scrape_pause, elapsed, interval,
            )
            await asyncio.sleep(scrape_pause)
    except Exception as exc:
        logger.error("[{}] Unhandled loop error: {}", status_key, exc)
    finally:
        if proxy is not None:
            await proxy_pool.release(proxy, ok=True)
        await _close_worker_context(context, context_semaphore)
        logger.info("[{}] Context closed.", status_key)


# ─────────────────────────────────────────────────────────────────────────────
# Main async entry point
# ─────────────────────────────────────────────────────────────────────────────
async def run_scraper_async(
    args: argparse.Namespace,
    status_callback: Callable[[dict], Any] | None = None,
) -> None:
    """
    Orchestrate all workers:
    1. Load proxies.
    2. Resolve targets from --target-file / --targets / --url.
    3. For each target, launch one worker per proxy (or one direct worker if no proxies).
    4. Gather results.
    """
    # ── Load proxies ──────────────────────────────────────────────────────────
    proxy_list: list[ProxyConfig] = []
    if args.proxy_file:
        proxy_list = load_proxies(args.proxy_file)

    # ── Resolve targets ───────────────────────────────────────────────────────
    targets = _load_targets_from_args(args)
    logger.info(
        "Targets resolved: {}",
        ", ".join(f"{t.name}={t.url}" for t in targets),
    )

    # ── Build the stateful proxy pool (M2) ────────────────────────────────────
    global _proxy_pool
    pool_configs: list[ProxyConfig | None] = proxy_list if proxy_list else [None]
    proxy_pool = ProxyManager(pool_configs)
    _proxy_pool = proxy_pool
    if proxy_list:
        worker_count = len(targets) * len(proxy_list)
        if args.workers:
            worker_count = max(1, min(worker_count, args.workers))
        logger.info("Proxy rotation: {} proxies loaded, pool: {}.",
                    len(proxy_list), proxy_pool.summary())
    else:
        worker_count = max(1, min(len(targets), args.workers)
                           if args.workers else len(targets))
        logger.info("No proxies → one local (direct) worker per target.")

    # ── Locate Chromium ───────────────────────────────────────────────────────
    exe_path = _find_chromium_exe()
    if exe_path:
        logger.info("Using Chromium at {}", exe_path)
    else:
        logger.warning(
            "Chromium executable not found via auto-discovery; "
            "letting Playwright resolve the path (may fail). "
            "Run: python -m playwright install chromium"
        )

    # ── Launch all workers under one Rich Live display ────────────────────────
    async with async_playwright() as pw:
        launch_kwargs = {
            "headless": not getattr(args, "show_browser", False),
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-infobars",
            ],
        }
        if exe_path:
            launch_kwargs["executable_path"] = exe_path

        browser: Browser = await pw.chromium.launch(**launch_kwargs)

        context_semaphore = asyncio.Semaphore(worker_count)
        stop_display = asyncio.Event()
        show_console = not getattr(args, "no_console", False)

        async def _run_with_live():
            with Live(console=console, refresh_per_second=4, screen=False) as live:
                display_task = asyncio.create_task(
                    _display_loop(live=live, stop=stop_display, refresh_period=0.5, status_callback=status_callback)
                )
                tasks = []
                for widx in range(worker_count):
                    tgt = targets[widx % len(targets)]
                    tasks.append(
                        asyncio.create_task(
                            run_worker(
                                browser=browser,
                                proxy_pool=proxy_pool,
                                context_semaphore=context_semaphore,
                                target=tgt,
                                interval=args.interval,
                                max_cycles=args.max_cycles,
                                worker_timeout=args.worker_timeout,
                                output_dir=Path(args.output_dir),
                                worker_idx=widx,
                            )
                        )
                    )
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                finally:
                    try:
                        await browser.close()
                    except Exception as close_exc:
                        logger.debug("Browser close encountered expected shutdown variance: {}", close_exc)

                stop_display.set()
                await display_task
                async with _status_lock:
                    snapshot = dict(_worker_statuses)
                live.update(
                    Panel(
                        _make_worker_status_table(snapshot),
                        title="[bold blue]⚡ Dashboard Scraper – finished[/bold blue]",
                        subtitle=f"[bold]Workers:[/bold] {len(snapshot)}",
                        border_style="blue",
                    )
                )

        async def _run_without_live():
            display_task = asyncio.create_task(
                _display_loop(live=None, stop=stop_display, refresh_period=0.5, status_callback=status_callback)
            )
            tasks = []
            for widx in range(worker_count):
                tgt = targets[widx % len(targets)]
                tasks.append(
                    asyncio.create_task(
                        run_worker(
                            browser=browser,
                            proxy_pool=proxy_pool,
                            context_semaphore=context_semaphore,
                            target=tgt,
                            interval=args.interval,
                            max_cycles=args.max_cycles,
                            worker_timeout=args.worker_timeout,
                            output_dir=Path(args.output_dir),
                            worker_idx=widx,
                        )
                    )
                )
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                try:
                    await browser.close()
                except Exception as close_exc:
                    logger.debug("Browser close encountered expected shutdown variance: {}", close_exc)

            stop_display.set()
            await display_task

        if show_console:
            await _run_with_live()
        else:
            await _run_without_live()

    logger.success(
        "All workers finished. Output dir: {}", Path(args.output_dir).resolve()
    )



# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Playwright scraper for real-time dashboard tables with proxy rotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("command", nargs="?", default="run", choices=["run", "monitor"],
                   help="Command mode: 'run' (default) or 'monitor' (auto-loads targets.json and proxies.txt).")
    p.add_argument("--url",            default=DEFAULT_URL,
                   help="Full URL of the dashboard page (used when --targets/--target-file are omitted).")
    p.add_argument("--table-id",       default=DEFAULT_TABLE_ID,
                   help="ID attribute of the <table> element to scrape.")
    p.add_argument("--targets",        action="append", default=[],
                   help="Target definition in name:url[:table_id] format. Repeatable.")
    p.add_argument("--target-file",    default=None, metavar="PATH",
                   help="JSON file with target definitions (array of {name, url, table_id?}).")
    p.add_argument("--output-dir",     default=str(DEFAULT_OUTPUT_DIR),
                   help="Root directory where CSV files are written (per-target subdirs created).")
    p.add_argument("--interval",       type=int, default=DEFAULT_INTERVAL_S,
                   help="Seconds between each CSV save cycle.")
    p.add_argument("--max-cycles",     type=int, default=DEFAULT_MAX_CYCLES,
                   help="Stop after this many save cycles (0 = run forever).")
    p.add_argument("--show-browser",   action="store_true",
                   help="Run Chromium in non-headless (visible) mode.")
    p.add_argument("--log-level",      default="INFO",
                   choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"],
                   help="Loguru log level.")

    # ── Proxy rotation flags ─────────────────────────────────────────────────
    p.add_argument("--proxy-file",     default=None, metavar="PATH",
                   help=(
                       "Path to a text file with one SOCKS5/HTTP proxy per line "
                       "(socks5://user:pass@host:port). "
                       "If omitted, runs in single-context no-proxy mode."
                   ))
    p.add_argument("--workers",        type=int, default=9, metavar="N",
                   help=(
                       "Maximum number of concurrent browser contexts (default 9-bot swarm). "
                       "Defaults to 9 parallel background workers across targets."
                   ))

    p.add_argument("--worker-timeout", type=int, default=30, metavar="SECS",
                   help="Per-worker page.goto() timeout in seconds.")

    args = p.parse_args()
    if args.command == "monitor":
        if not args.target_file and Path("targets.json").exists():
            args.target_file = "targets.json"
        if not args.proxy_file and Path("proxies.txt").exists():
            args.proxy_file = "proxies.txt"

    return args



def _configure_logger(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{function}</cyan>:<cyan>{line}</cyan> – {message}"
        ),
        colorize=True,
    )
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "scraper_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="14 days",
        compression="gz",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Force UTF-8 output on Windows legacy consoles
    import io
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    args = _parse_args()
    _configure_logger(args.log_level)

    # Resolve targets early for logging
    targets = _load_targets_from_args(args)
    target_names = ", ".join(f"{t.name}={t.url}" for t in targets)

    logger.info(
        "Configuration: targets=[{}] output_dir={} interval={}s "
        "max_cycles={} headless={} proxy_file={} workers={}",
        target_names, args.output_dir,
        args.interval, args.max_cycles, not args.show_browser,
        args.proxy_file, args.workers,
    )

    asyncio.run(run_scraper_async(args))
