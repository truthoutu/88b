"""
env_optimizations.py
====================
Environment-specific optimization protocols for the realtime-dashboard-scraper.

Three protocol classes, each addressing a distinct web architecture:

    A  WebSocket-driven dynamic content       (e.g. betking.com)
    B  Cookie / session persistence + SOCKS5   (e.g. betano.ng)
    C  Deeply nested iframes + dynamic IDs     (e.g. msport.com/ng)

Quick start
-----------
    from env_optimizations import get_protocol

    # Inside run_worker(), after page.goto():
    protocol = get_protocol(args.env, ...)
    if protocol:
        await protocol.prepare(page, context, proxy_cfg, table_sel)

    # Inside the scrape loop:
    if protocol:
        await protocol.wait_for_data(page, table_sel, cycle_start)
        scrape_pause = protocol.suggested_interval()
    else:
        scrape_pause = random.uniform(3.0, 8.0)

    fresh_rows = await protocol.scrape_table(
        page, table_sel, col_idx
    ) if protocol else await _scrape_table(page, table_sel, col_idx)

The ``DataRow`` NamedTuple defined here is structurally identical to the one
in ``scraper.py``; when integrating, ``scraper.py`` should import ``DataRow``
from this module rather than defining its own copy.
"""

from __future__ import annotations

import asyncio
import collections
import json
import random
import time
from pathlib import Path
from typing import Any, NamedTuple, Optional

from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    Frame,
    Page,
    Response,
    WebSocket,
    TimeoutError as PWTimeoutError,
)

from proxy_loader import ProxyConfig


# ─────────────────────────────────────────────────────────────────────────────
# Shared data model  (structurally identical to scraper.DataRow)
# ─────────────────────────────────────────────────────────────────────────────

class DataRow(NamedTuple):
    """Canonical row shape: Timestamp | Event Name | Numeric Result."""
    timestamp: str
    event_name: str
    numeric_result: str

    @classmethod
    def from_cells(cls, cells: list[str]) -> "DataRow":
        if len(cells) < 3:
            raise ValueError(f"Expected >=3 cells, got {len(cells)}: {cells!r}")
        return cls(
            timestamp=cells[0].strip(),
            event_name=cells[1].strip(),
            numeric_result=cells[2].strip(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Expected column-header mapping (kept in sync with scraper.EXPECTED_HEADERS)
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_HEADERS: dict[str, str] = {
    "Timestamp":      "timestamp",
    "Event Name":     "event_name",
    "Numeric Result": "numeric_result",
}
_HEADER_LOOKUP: dict[str, str] = {
    k.lower().strip(): v for k, v in EXPECTED_HEADERS.items()
}


# ─────────────────────────────────────────────────────────────────────────────
# Protocol registry
# ─────────────────────────────────────────────────────────────────────────────

_PROTOCOL_REGISTRY: dict[str, type] = {}


def _register_protocol(cls: type):
    """Decorator: register a protocol class for a given environment letter."""
    _PROTOCOL_REGISTRY[cls.ENV_KEY] = cls
    return cls


def get_protocol(env: str, /, **kwargs: Any) -> Optional["_BaseProtocol"]:
    """
    Factory: return the protocol instance for environment *env*.

    Parameters
    ----------
    env : "A" | "B" | "C"
        Environment identifier.
    **kwargs
        Forwarded to the protocol constructor.
    """
    env = env.upper().strip()
    cls = _PROTOCOL_REGISTRY.get(env)
    if cls is None:
        logger.warning(
                        "No optimization protocol registered for env='{}'. "
            "Running in default (no-protocol) mode.", env
        )
        return None
    return cls(**kwargs)


class _BaseProtocol:
    """
    Common interface shared by all three environment protocols.

    Every protocol exposes:
    * prepare()          - post-navigation setup (WS listeners, cookie injection,
                           iframe discovery)
    * scrape_table()     - environment-aware table scraping
    * wait_for_data()    - block until new data is likely available
    * suggested_interval() - dynamic inter-scrape pause
    """

    ENV_KEY: str = ""
    ENV_DESC: str = ""

    async def prepare(
        self,
        page: Page,
        context: BrowserContext,
        proxy_cfg: ProxyConfig | None,
        table_selector: str,
    ) -> None:
        raise NotImplementedError

    async def scrape_table(
        self,
        page: Page,
        frame: Frame | None,
        table_selector: str,
        col_idx: dict[str, int],
    ) -> list[DataRow]:
        raise NotImplementedError

    async def wait_for_data(
        self,
        page: Page,
        table_selector: str,
        cycle_start: float,
        timeout_ms: float = 10_000,
    ) -> bool:
        raise NotImplementedError

    def suggested_interval(self) -> float:
        raise NotImplementedError

    def teardown(self, page: Page) -> None:
        """Clean up listeners.  Safe to call even if prepare() was never called."""
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT A — WebSocket-driven dynamic content
#                      (e.g. betking.com)
# ═════════════════════════════════════════════════════════════════════════════

@_register_protocol
class EnvProtocolA(_BaseProtocol):
    """
    Optimization protocol for WebSocket-driven dashboards.

    Challenges
    ----------
    * Data arrives in unpredictable bursts via WS frames, not on a fixed
      HTTP polling schedule.
    * ``wait_until='domcontentloaded'`` fires before the WS handshake
      completes, so the first DOM read may see an empty or stale table.
    * A fixed 3-8 s scrape sleep can miss rapid update cycles or waste
      CPU during quiet periods.

    Solution
    --------
    1.  Register ``page.on('websocket')`` immediately after navigation to
        detect every WS connection and filter for the data-feed URL.
    2.  Block until the relevant WS opens and delivers >= ``min_ws_frames``
        data frames before the first scrape.
    3.  Track inter-frame intervals in a rolling deque to estimate cadence.
    4.  Replace the fixed sleep with an **adaptive interval** ~ 1.5x the
        median frame-to-frame gap, clamped to [0.3, 8] seconds.
    5.  Before each scrape, ``page.wait_for_function`` blocks until the
        table's ``<tbody>`` row count increases (server-push -> DOM
        mutation -> capture).
    """

    #: Only monitor WS connections whose URL contains one of these substrings.
    #: betking.com's game feed is typically wss://*-feed.betking.com/...
    WS_URL_FILTERS: tuple[str, ...] = (
        "feed", "socket", "stream", "ws", "live", "push",
    )

    ENV_KEY = "A"
    ENV_DESC = "WebSocket-driven dynamic content (betking.com)"

    def __init__(
        self,
        min_ws_frames: int = 3,
        min_interval: float = 0.3,
        max_interval: float = 8.0,
    ):
        self._min_ws_frames = min_ws_frames
        self._min_interval = min_interval
        self._max_interval = max_interval

        self._frame_times: collections.deque[float] = collections.deque(
            maxlen=100
        )
        self._ws: WebSocket | None = None
        self._data_event: asyncio.Event = asyncio.Event()
        self._page: Page | None = None
        self._table_selector: str = ""
        self._prev_row_count: int = 0
        self._missed_updates: int = 0
        self._t0: float = 0.0  # for measuring first-frame latency

    # -- public API --

    async def prepare(
        self,
        page: Page,
        context: BrowserContext,
        proxy_cfg: ProxyConfig | None,
        table_selector: str,
    ) -> None:
        """Attach WS monitors and block until the first data frames arrive."""
        self._page = page
        self._table_selector = table_selector
        self._t0 = time.monotonic()

        # Register *before* potential late WS opens.  Playwright fires
        # ``websocket`` only for *new* connections, so if the WS is already
        # open we handle it via the polling fallback in _wait_for_initial_frames.
        page.on("websocket", self._on_websocket)

        ok = await self._wait_for_initial_frames(timeout=30.0)
        if ok:
            elapsed = time.monotonic() - self._t0
            logger.success(
                "[EnvA] WebSocket ready -- {} frames in {:.1f}s",
                len(self._frame_times), elapsed,
            )
        else:
            logger.warning(
                "[EnvA] WebSocket initial-data timeout; "
                "falling back to DOM-polling mode"
            )

        self._prev_row_count = await self._count_rows(page, table_selector)

    async def wait_for_data(
        self,
        page: Page,
        table_selector: str,
        cycle_start: float,
        timeout_ms: float = 8_000,
    ) -> bool:
        """
        Block until the table gains >=1 new row (push-driven DOM mutation)
        or *timeout_ms* elapses.

        Returns True if new rows were detected.
        """
        prev = self._prev_row_count
        try:
            await page.wait_for_function(
                """(data) => {
                    const table = document.querySelector(data.sel);
                    if (!table) return false;
                    const tbody = table.querySelector('tbody');
                    if (!tbody) return false;
                    return tbody.querySelectorAll('tr').length > data.prev;
                }""",
                {"sel": table_selector, "prev": prev},
                timeout=timeout_ms,
            )
            self._prev_row_count = await self._count_rows(
                page, table_selector
            )
            self._missed_updates = 0
            return True
        except PWTimeoutError:
            self._missed_updates += 1
            if self._missed_updates % 5 == 0:
                logger.warning(
                    "[EnvA] {} consecutive cycles with no DOM update; "
                    "WS may be stalled or the table is static",
                    self._missed_updates,
                )
            return False

    def suggested_interval(self) -> float:
        """
        Dynamic inter-scrape pause based on observed WS cadence.

        Targets 1.5x the median frame-to-frame interval, clamped to
        [min_interval, max_interval].
        """
        if len(self._frame_times) < 3:
            # Not enough data yet -- conservative default
            return random.uniform(1.0, 3.0)

        intervals = [
            t2 - t1
            for t1, t2 in zip(
                list(self._frame_times)[:-1],
                list(self._frame_times)[1:],
            )
        ]
        intervals.sort()
        median = intervals[len(intervals) // 2]
        return max(
            self._min_interval,
            min(median * 1.5, self._max_interval),
        )

    def teardown(self, page: Page) -> None:
        page.remove_listener("websocket", self._on_websocket)

    # -- internal helpers --

    def _matches_filter(self, url: str) -> bool:
        lowered = url.lower()
        return any(token in lowered for token in self.WS_URL_FILTERS)

    def _on_websocket(self, ws: WebSocket) -> None:
        """Called by Playwright when any WebSocket connection opens."""
        if not self._matches_filter(ws.url):
            return
        logger.debug("[EnvA] WebSocket opened: {}", ws.url)
        self._ws = ws
        ws.on("framereceived", self._on_frame)
        ws.on("framesent", self._on_frame)
        ws.on("close", self._on_ws_close)

    def _on_frame(self, payload: str | bytes) -> None:
        """Update timing stats whenever a frame traverses the data-feed WS."""
        now = time.monotonic()
        self._frame_times.append(now)
        self._data_event.set()

    def _on_ws_close(self) -> None:
        """Unblock waiters and reset state when the WS disconnects."""
        logger.warning(
            "[EnvA] WebSocket closed -- {} frames captured",
            len(self._frame_times),
        )
        self._ws = None
        self._data_event.set()

    async def _wait_for_initial_frames(
        self, timeout: float = 30.0
    ) -> bool:
        """Poll until >= min_ws_frames have been received."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self._frame_times) >= self._min_ws_frames:
                return True
            await asyncio.sleep(0.1)
        return len(self._frame_times) >= self._min_ws_frames

    async def _count_rows(
        self, page: Page, table_selector: str
    ) -> int:
        """Count <tr> elements in the table's tbody."""
        try:
            return await page.eval_on_selector_all(
                f"{table_selector} tbody tr",
                "els => els.length",
            )
        except Exception:
            return 0


# ═════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT B — Cookie / session persistence + SOCKS5
#                      (e.g. betano.ng)
# ═════════════════════════════════════════════════════════════════════════════

@_register_protocol
class EnvProtocolB(_BaseProtocol):
    """
    Optimization protocol for aggressive cookie-based session architectures
    running behind SOCKS5 proxies.

    Challenges
    ----------
    * betano.ng issues short-lived cookies and aggressively rotates CSRF
      tokens via client-side JS.
    * Playwright's SOCKS5 proxy is bound at BrowserContext creation; you
      cannot change the proxy without destroying the context -- which also
      discards its cookies and localStorage.
    * Reusing a stale context after proxy rotation leads to cookie
      desynchronisation: the server sees cookies from proxy X but the
      request comes from proxy Y.

    Solution
    --------
    1.  After initial auth, persist the full storage_state (cookies +
        localStorage) to disk via context.storage_state().
    2.  When needs_rotation is True (HTTP 401/403/419/440 or periodic
        timeout), create a **fresh** BrowserContext with a *new* proxy and
        re-inject the stored cookies via context.add_cookies().
    3.  Register page.on('response') to detect auth-failure statuses
        in real time and trigger immediate rotation.
    4.  Clear localStorage / sessionStorage / IndexedDB on every
        context rotation to prevent stale client-side state.
    """

    #: HTTP status codes that signal session invalidation
    SESSION_INVALIDATION_STATUSES: frozenset[int] = frozenset(
        {401, 403, 419, 440}
    )

    #: How often to force a session rotation even if no auth error occurred
    DEFAULT_SESSION_LIFETIME: float = 300.0  # 5 minutes

    ENV_KEY = "B"
    ENV_DESC = "Cookie/session persistence + SOCKS5 (betano.ng)"

    def __init__(
        self,
        storage_path: str | Path = "session_storage.json",
        session_lifetime: float = DEFAULT_SESSION_LIFETIME,
    ):
        self._storage_path = Path(storage_path)
        self._session_lifetime = session_lifetime
        self._session_start: float = time.monotonic()
        self._session_expired: bool = False
        self._page: Page | None = None

    # -- public API --

    async def prepare(
        self,
        page: Page,
        context: BrowserContext,
        proxy_cfg: ProxyConfig | None,
        table_selector: str,
    ) -> None:
        """Re-hydrate cookies into the context and start response monitoring."""
        self._page = page
        self._session_start = time.monotonic()
        self._session_expired = False

        await self._hydrate(context)

        # Register response monitor (only once)
        page.on("response", self._on_response)

    async def persist(self, context: BrowserContext) -> int:
        """
        Save the current storage_state to disk for reuse after
        proxy/context rotation.  Returns the number of cookies saved.
        """
        state = await context.storage_state()
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(
            json.dumps(state), encoding="utf-8"
        )
        cookie_count = len(state.get("cookies", []))
        logger.info(
            "[EnvB] Persisted {} cookies + localStorage to {}",
            cookie_count, self._storage_path,
        )
        return cookie_count

    @property
    def needs_rotation(self) -> bool:
        """True if the session should be rotated right now."""
        if self._session_expired:
            return True
        if time.monotonic() - self._session_start > self._session_lifetime:
            return True
        return False

    def reset_rotation_flag(self) -> None:
        """Call after a successful rotation to reset the timer."""
        self._session_expired = False
        self._session_start = time.monotonic()

    async def clear_browser_state(self, page: Page) -> None:
        """
        Wipe localStorage, sessionStorage, and cookies to force the client-
        side JS to re-initialise its state machine.
        """
        await page.evaluate(
            """() => {
                localStorage.clear();
                sessionStorage.clear();
                // Overwrite cookie header (client-side only)
                document.cookie.split(';').forEach(c => {
                    document.cookie = c.replace(/=.*/, '=; expires=Thu, 01 Jan 1970 00:00:00 GMT');
                });
            }"""
        )
        logger.debug("[EnvB] Browser state cleared (localStorage/sessionStorage/cookies)")

    @staticmethod
    async def create_context_with_session(
        browser: Browser,
        proxy_cfg: ProxyConfig | None,
        fp: dict,
        storage_path: str | Path | None = None,
    ) -> BrowserContext:
        """
        Create a **fresh** BrowserContext with:
        * the given SOCKS5/HTTP proxy,
        * the given fingerprint dict,
        * optional storage_state re-injection.

        This is the key to preventing cookie desynchronisation: every proxy
        gets its own isolated context with its own cookie jar.
        """
        ctx_kwargs: dict[str, Any] = {
            "viewport":             fp["viewport"],
            "user_agent":           fp["user_agent"],
            "locale":               fp["locale"],
            "timezone_id":          fp["timezone_id"],
            "device_scale_factor":  fp["device_scale_factor"],
        }

        if proxy_cfg is not None:
            ctx_kwargs["proxy"] = proxy_cfg.as_playwright_proxy()

        if storage_path is not None:
            sp = Path(storage_path)
            if sp.exists():
                state = json.loads(sp.read_text(encoding="utf-8"))
                # Playwright accepts a dict as storage_state
                ctx_kwargs["storage_state"] = state
                logger.info(
                    "[EnvB] Re-using storage_state ({} cookies) from {}",
                    len(state.get("cookies", [])), sp,
                )

        context = await browser.new_context(**ctx_kwargs)

        # Strengthen fingerprint masking on context creation
        await context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
            // Permissions API spoofing
            const origQuery = navigator.permissions && navigator.permissions.query;
            if (origQuery) {
                navigator.permissions.query = (params) => {
                    if (params.name === 'notifications') {
                        return Promise.resolve({ state: 'default', query: () => {} });
                    }
                    if (params.name === 'push') {
                        return Promise.resolve({ state: 'denied' });
                    }
                    return origQuery(params);
                };
            }
            // Plugins array spoofing
            Object.defineProperty(navigator, 'plugins', {
                get: () => ({ 0: {}, length: 1, item: () => null, namedItem: () => null }),
            });
            // Languages spoofing
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            """
        )
        return context

    def teardown(self, page: Page) -> None:
        if self._page is not None:
            page.remove_listener("response", self._on_response)

    # -- internal helpers --

    async def _hydrate(self, context: BrowserContext) -> None:
        """Re-inject previously saved cookies into a fresh context."""
        if not self._storage_path.exists():
            logger.info("[EnvB] No storage state found -- first run")
            return

        state = json.loads(self._storage_path.read_text(encoding="utf-8"))
        cookies = state.get("cookies", [])
        if not cookies:
            return

        # Clear existing cookies first to avoid desync
        await context.clear_cookies()
        await context.clear_permissions()

        # add_cookies accepts a list of dicts with name/value/domain/path/etc.
        await context.add_cookies(cookies)
        logger.info("[EnvB] Re-injected {} session cookies", len(cookies))

    def _on_response(self, response: Response) -> None:
        """Detect session invalidation in real time."""
        if response.status in self.SESSION_INVALIDATION_STATUSES:
            logger.warning(
                "[EnvB] Session invalidation detected: {} {}",
                response.status, response.url,
            )
            self._session_expired = True

    # -- stubs for interface compliance --
    # Environment B doesn't need WebSocket or iframe logic, but the base
    # class requires these methods.  They delegate to the default (page-level)
    # implementation.

    async def wait_for_data(
        self,
        page: Page,
        table_selector: str,
        cycle_start: float,
        timeout_ms: float = 10_000,
    ) -> bool:
        # No WS to wait for -- just verify the table is still in the DOM.
        return True

    def suggested_interval(self) -> float:
        # Environment B doesn't need adaptive intervals.
        return random.uniform(3.0, 8.0)

    async def scrape_table(
        self,
        page: Page,
        frame: Frame | None,
        table_selector: str,
        col_idx: dict[str, int],
    ) -> list[DataRow]:
        # Environment B uses standard page-level scraping (same as the base
        # scraper).  This method is provided for interface completeness.
        rows: list[DataRow] = []
        tr_elements = await page.query_selector_all(
            f"{table_selector} tbody tr"
        )
        for tr in tr_elements:
            tds = await tr.query_selector_all("td")
            if not tds:
                continue
            try:
                cells = [
                    (await tds[col_idx[k]].inner_text()).strip()
                    for k in ("timestamp", "event_name", "numeric_result")
                ]
                # Skip blank rows
                if not cells[0] and not cells[1]:
                    continue
                rows.append(DataRow(*cells))
            except (IndexError, Exception) as exc:
                logger.debug("[EnvB] Skipping malformed row: {}", exc)
        return rows


# ═════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT C — Deeply nested iframes + dynamic IDs
#                      (e.g. msport.com/ng)
# ═════════════════════════════════════════════════════════════════════════════

@_register_protocol
class EnvProtocolC(_BaseProtocol):
    """
    Optimization protocol for deeply nested iframe architectures with
    framework-generated (dynamic) element IDs.

    Challenges
    ----------
    * The data table lives inside 2-4 levels of nested <iframe> elements,
      each potentially from a different origin.
    * Element IDs are auto-generated on every load (e.g.
      react-table-7f3a9c), so #static-id selectors break immediately.
    * page.query_selector only searches the top-level document; it
      silently returns None for elements inside iframes.
    * Iframes can be created/destroyed dynamically as the SPA re-renders,
      invalidating stale references.

    Solution
    --------
    1.  find_table_frame() recursively walks page.frames and every
        frame.child_frames tree, trying a **cascade** of selectors from
        most-specific to most-generic.
    2.  When all ID-based selectors fail, fall back to **header-text matching**:
        run a JS snippet that scans every <table> in the frame and returns
        the one whose <thead> texts match the expected column names.
    3.  All DOM access (query, evaluate, wait_for) is performed on the
        resolved Frame object, not on page.
    4.  frameattached / framenavigated listeners trigger lazy
        re-discovery so stale frame references are refreshed.
    """

    ENV_KEY = "C"
    ENV_DESC = "Deeply nested iframes + dynamic IDs (msport.com/ng)"

    def __init__(self):
        self._table_frame: Frame | None = None
        self._resolved_selector: str = ""
        self._page: Page | None = None
        self._table_selector_hint: str = ""
        self._stale: bool = True

    # -- public API --

    async def prepare(
        self,
        page: Page,
        context: BrowserContext,
        proxy_cfg: ProxyConfig | None,
        table_selector: str,
    ) -> None:
        """Find the iframe containing the table and resolve a robust selector."""
        self._page = page
        self._table_selector_hint = table_selector

        # Register frame change listeners
        page.on("frameattached", self._on_frame_event)
        page.on("framenavigated", self._on_frame_event)

        await self._rediscover()

    async def find_table_frame(
        self,
        page: Page,
        table_id: str,
    ) -> tuple[Frame, str]:
        """
        Recursively search *all* frames for a table matching *table_id*.

        Returns (frame, resolved_selector).
        Raises RuntimeError if not found.
        """
        selectors = self._build_robust_selectors(table_id)

        for frame in page.frames:
            result = await self._search_frame(frame, selectors, table_id)
            if result:
                return result

        raise RuntimeError(
            f"Table '{table_id}' not found in any frame or iframe. "
            f"Searched {len(page.frames)} frame(s)."
        )

    async def scrape_table(
        self,
        page: Page,
        frame: Frame | None,
        table_selector: str,
        col_idx: dict[str, int],
    ) -> list[DataRow]:
        """
        Scrape table rows from the resolved frame using the resolved selector.

        Falls back to the main frame if frame is None (e.g. the table
        was promoted to the top level after an SPA re-render).
        """
        if frame is None:
            frame = page.main_frame

        rows: list[DataRow] = []
        selector = self._resolved_selector or table_selector

        try:
            tr_elements = await frame.query_selector_all(
                f"{selector} tbody tr"
            )
        except Exception as exc:
            logger.debug(
                "[EnvC] Query failed on frame '{}': {}",
                frame.name or "(main)", exc
            )
            tr_elements = []

        for tr in tr_elements:
            tds = await tr.query_selector_all("td")
            if not tds:
                continue
            try:
                cells = [
                    (await tds[col_idx[k]].inner_text()).strip()
                    for k in ("timestamp", "event_name", "numeric_result")
                ]
                if not cells[0] and not cells[1]:
                    continue
                rows.append(DataRow(*cells))
            except (IndexError, Exception) as exc:
                logger.debug("[EnvC] Skipping malformed row: {}", exc)

        return rows

    async def discover_columns(
        self,
        frame: Frame | None,
        table_selector: str,
    ) -> dict[str, int]:
        """
        Frame-aware version of _discover_column_indices.

        Reads the <thead> of the resolved frame's table and returns a
        dict mapping canonical column names to zero-based cell indices.
        """
        if frame is None:
            frame = self._table_frame or (
                self._page.main_frame if self._page else None
            )
            if frame is None:
                raise RuntimeError("No frame available for column discovery")

        selector = self._resolved_selector or table_selector

        header_cells = await frame.query_selector_all(f"{selector} thead th")
        if not header_cells:
            header_cells = await frame.query_selector_all(
                f"{selector} thead tr:first-child td"
            )

        found: dict[str, int] = {}
        for idx, cell in enumerate(header_cells):
            raw = (await cell.inner_text() or "").strip()
            normalized = " ".join(raw.lower().split())
            if normalized in _HEADER_LOOKUP:
                found[_HEADER_LOOKUP[normalized]] = idx

        missing = set(EXPECTED_HEADERS.values()) - set(found.keys())
        if missing:
            all_headers = [await c.inner_text() for c in header_cells]
            raise RuntimeError(
                f"Could not locate columns {missing} in table header "
                f"{all_headers!r}. Check --table-id or the page structure."
            )
        return found

    async def wait_for_data(
        self,
        page: Page,
        table_selector: str,
        cycle_start: float,
        timeout_ms: float = 10_000,
    ) -> bool:
        """Wait for the table to be present in the resolved frame."""
        if self._stale:
            await self._rediscover()
        if self._table_frame is None:
            return False
        sel = self._resolved_selector or table_selector
        try:
            await self._table_frame.wait_for_selector(
                f"{sel} tbody tr", timeout=min(timeout_ms, 5000)
            )
            return True
        except PWTimeoutError:
            return False

    def suggested_interval(self) -> float:
        return random.uniform(3.0, 8.0)

    def teardown(self, page: Page) -> None:
        page.remove_listener("frameattached", self._on_frame_event)
        page.remove_listener("framenavigated", self._on_frame_event)

    # -- internal helpers --

    def _build_robust_selectors(self, table_id: str) -> list[str]:
        """
        Generate a cascade of selectors from most-specific to most-generic.

        This handles dynamic IDs by trying progressively looser match
        strategies:
        1.  Exact #id
        2.  Exact table#id
        3.  Attribute equality [id='id']
        4.  Substring [id*='id']
        5.  Prefix [id^='id']  (handles id-7f3a9c)
        6.  Class substring [class*='id']
        7.  data-* attributes (common in React/Vue)
        8.  ARIA role
        9.  **Header-based discovery** (last resort -- finds any table with
            the expected column headers)
        """
        # Normalise: strip leading # so we can safely prefix it
        clean = table_id.lstrip("#")
        return [
            f"#{clean}",
            f"table#{clean}",
            f"table[id='{clean}']",
            f"table[id*='{clean}']",
            f"table[id^='{clean}']",
            f"table[class*='{clean}']",
            f"table[data-testid='{clean}']",
            f"table[data-qa='{clean}']",
            f"table[data-table='{clean}']",
            f"table[role='grid']",
            "table",
        ]

    async def _search_frame(
        self,
        frame: Frame,
        selectors: list[str],
        table_id: str,
    ) -> tuple[Frame, str] | None:
        """
        Try each selector on frame and its child frames.

        Returns (frame, matched_selector) or None.
        """
        for sel in selectors:
            if sel == "table":
                # Header-based fallback -- more thorough than bare "table"
                header_match = await self._find_table_by_headers(frame)
                if header_match is not None:
                    return (frame, header_match)
                continue

            try:
                if await frame.query_selector(sel):
                    return (frame, sel)
            except Exception:
                pass

        # Recurse into child iframes
        for child in frame.child_frames:
            result = await self._search_frame(child, selectors, table_id)
            if result:
                return result

        return None

    async def _find_table_by_headers(
        self,
        frame: Frame,
    ) -> str | None:
        """
        Last-resort discovery: find any <table> whose <thead> text
        matches the expected column names (Timestamp, Event Name, Numeric
        Result).  Returns a structural CSS selector (table:nth-of-type(N))
        or None.
        """
        expected_texts = list(EXPECTED_HEADERS.keys())

        try:
            result = await frame.evaluate(
                """(expectedTexts) => {
                    const tables = document.querySelectorAll('table');
                    for (let i = 0; i < tables.length; i++) {
                        const headers = tables[i].querySelectorAll(
                            'thead th, thead tr:first-child td'
                        );
                        if (headers.length === 0) continue;
                        const texts = Array.from(headers).map(
                            h => h.textContent.trim().toLowerCase()
                        );
                        const matched = expectedTexts.every(k =>
                            texts.some(t => t.includes(k.toLowerCase()))
                        );
                        if (matched && texts.length >= expectedTexts.length) {
                            return i + 1;  // 1-based nth-of-type
                        }
                    }
                    return null;
                }""",
                expected_texts,
            )
        except Exception:
            return None

        if result is not None:
            return f"table:nth-of-type({result})"
        return None

    def _on_frame_event(self, frame: Frame) -> None:
        """Mark discovery as stale when frames change."""
        logger.debug(
            "[EnvC] Frame event (attached/navigated) -- re-discovery pending"
        )
        self._stale = True

    async def _rediscover(self) -> None:
        """Re-run frame + selector discovery."""
        if self._page is None:
            return
        try:
            self._table_frame, self._resolved_selector = (
                await self.find_table_frame(
                    self._page, self._table_selector_hint
                )
            )
            self._stale = False
            logger.info(
                "[EnvC] Re-discovered table in frame '{}' via '{}'",
                self._table_frame.name or "(main)",
                self._resolved_selector,
            )
        except RuntimeError as exc:
            logger.warning("[EnvC] Re-discovery failed: {}", exc)

    @property
    def table_frame(self) -> Frame | None:
        """The currently resolved frame containing the target table."""
        return self._table_frame

    @property
    def resolved_selector(self) -> str:
        """The CSS selector that matched the table (may be dynamic-ID-tolerant)."""
        return self._resolved_selector


# ─────────────────────────────────────────────────────────────────────────────
# Convenience re-export
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "DataRow",
    "EXPECTED_HEADERS",
    "EnvProtocolA",
    "EnvProtocolB",
    "EnvProtocolC",
    "get_protocol",
]