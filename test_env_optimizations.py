"""
test_env_optimizations.py
=========================
Unit tests for the three environment optimization protocols.

These tests exercise the *logic* of each protocol (registry, intervals,
selector cascades, session rotation flags, frame discovery) without
launching a browser or hitting live sites.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import env_optimizations as eo
from env_optimizations import (
    DataRow,
    EnvProtocolA,
    EnvProtocolB,
    EnvProtocolC,
    get_protocol,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_proxy(label="p00", server="socks5://127.0.0.1:1080"):
    return eo.ProxyConfig(server=server, username=None, password=None, label=label)


def _make_page_mock():
    page = MagicMock()
    page.on = MagicMock()
    page.remove_listener = MagicMock()
    return page


def _make_context_mock():
    ctx = MagicMock()
    ctx.storage_state = AsyncMock(return_value={"cookies": [{"name": "a", "value": "b"}]})
    ctx.add_cookies = AsyncMock()
    ctx.clear_cookies = AsyncMock()
    ctx.clear_permissions = AsyncMock()
    return ctx


def _make_browser_mock():
    browser = MagicMock()
    browser.new_context = AsyncMock()
    return browser


# ─────────────────────────────────────────────────────────────────────────────
# Registry tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_get_protocol_a(self):
        p = get_protocol("A")
        assert isinstance(p, EnvProtocolA)

    def test_get_protocol_b(self):
        p = get_protocol("B")
        assert isinstance(p, EnvProtocolB)

    def test_get_protocol_c(self):
        p = get_protocol("C")
        assert isinstance(p, EnvProtocolC)

    def test_get_protocol_unknown_returns_none(self):
        assert get_protocol("Z") is None

    def test_get_protocol_case_insensitive(self):
        assert isinstance(get_protocol("a"), EnvProtocolA)
        assert isinstance(get_protocol("  c  "), EnvProtocolC)


# ─────────────────────────────────────────────────────────────────────────────
# DataRow tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataRow:
    def test_from_cells_ok(self):
        r = DataRow.from_cells(["2026-01-01", "CPU_SPIKE", "87.5"])
        assert r == DataRow("2026-01-01", "CPU_SPIKE", "87.5")

    def test_from_cells_strips(self):
        r = DataRow.from_cells(["  2026-01-01  ", "  CPU_SPIKE  ", "  87.5  "])
        assert r == DataRow("2026-01-01", "CPU_SPIKE", "87.5")

    def test_from_cells_too_few_raises(self):
        with pytest.raises(ValueError):
            DataRow.from_cells(["2026-01-01", "CPU_SPIKE"])


# ─────────────────────────────────────────────────────────────────────────────
# Environment A — WebSocket-driven
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvProtocolA:
    def test_suggested_interval_default(self):
        p = EnvProtocolA()
        for _ in range(20):
            iv = p.suggested_interval()
            assert 0.3 <= iv <= 8.0

    def test_suggested_interval_adaptive(self):
        p = EnvProtocolA()
        now = time.monotonic()
        # Simulate 10 frames at 0.5 s intervals
        for i in range(10):
            p._frame_times.append(now + i * 0.5)
        # Median interval is 0.5, 1.5x = 0.75
        iv = p.suggested_interval()
        assert 0.3 <= iv <= 8.0
        assert 0.4 <= iv <= 1.5

    @pytest.mark.asyncio
    async def test_wait_for_initial_frames_times_out(self):
        p = EnvProtocolA(min_ws_frames=5)
        ok = await p._wait_for_initial_frames(timeout=0.1)
        assert ok is False

    @pytest.mark.asyncio
    async def test_wait_for_initial_frames_succeeds(self):
        p = EnvProtocolA(min_ws_frames=2)
        # Simulate frames arriving via the on_frame callback
        p._on_frame("payload1")
        p._on_frame("payload2")
        ok = await p._wait_for_initial_frames(timeout=1.0)
        assert ok is True

    def test_matches_filter(self):
        p = EnvProtocolA()
        assert p._matches_filter("wss://feed.betking.com/ws")
        assert p._matches_filter("wss://live.example.com/socket")
        assert not p._matches_filter("https://example.com/page")

    @pytest.mark.asyncio
    async def test_prepare_registers_listener(self):
        page = _make_page_mock()
        p = EnvProtocolA()
        await p.prepare(page, MagicMock(), _make_proxy(), "#table")
        page.on.assert_called_with("websocket", p._on_websocket)

    @pytest.mark.asyncio
    async def test_teardown_removes_listener(self):
        page = _make_page_mock()
        p = EnvProtocolA()
        await p.prepare(page, MagicMock(), _make_proxy(), "#table")
        p.teardown(page)
        page.remove_listener.assert_called_with("websocket", p._on_websocket)

    @pytest.mark.asyncio
    async def test_count_rows_mock(self):
        page = MagicMock()
        page.eval_on_selector_all = AsyncMock(return_value=5)
        p = EnvProtocolA()
        count = await p._count_rows(page, "#metrics-table")
        assert count == 5
        page.eval_on_selector_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_data_no_update(self):
        page = MagicMock()
        from playwright.async_api import TimeoutError as PWTimeoutError
        page.wait_for_function = AsyncMock(side_effect=PWTimeoutError("timeout"))
        p = EnvProtocolA()
        result = await p.wait_for_data(page, "#table", time.monotonic(), timeout_ms=100)
        assert result is False
        assert p._missed_updates == 1


# ─────────────────────────────────────────────────────────────────────────────
# Environment B — Cookie/session persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvProtocolB:
    def test_needs_rotation_initial(self):
        p = EnvProtocolB()
        assert p.needs_rotation is False

    @pytest.mark.asyncio
    async def test_persist_saves_cookies(self, tmp_path):
        ctx = _make_context_mock()
        p = EnvProtocolB(storage_path=tmp_path / "session.json")
        count = await p.persist(ctx)
        assert count == 1
        saved = json.loads((tmp_path / "session.json").read_text())
        assert len(saved["cookies"]) == 1

    @pytest.mark.asyncio
    async def test_hydrate_no_file(self):
        ctx = _make_context_mock()
        p = EnvProtocolB(storage_path="/nonexistent/session.json")
        # Should not raise
        await p._hydrate(ctx)

    @pytest.mark.asyncio
    async def test_hydrate_with_cookies(self, tmp_path):
        # Write a mock storage state
        state = {"cookies": [{"name": "sid", "value": "abc", "domain": ".test.com"}]}
        path = tmp_path / "session.json"
        path.write_text(json.dumps(state))
        ctx = _make_context_mock()
        p = EnvProtocolB(storage_path=path)
        await p._hydrate(ctx)
        ctx.clear_cookies.assert_called_once()
        ctx.clear_permissions.assert_called_once()
        ctx.add_cookies.assert_called_once_with(state["cookies"])

    @pytest.mark.asyncio
    async def test_on_response_detects_auth_error(self):
        p = EnvProtocolB()
        resp = MagicMock()
        resp.status = 401
        resp.url = "https://betano.ng/api/data"
        p._on_response(resp)
        assert p._session_expired is True

    @pytest.mark.asyncio
    async def test_on_response_ignores_ok(self):
        p = EnvProtocolB()
        resp = MagicMock()
        resp.status = 200
        resp.url = "https://betano.ng/api/data"
        p._on_response(resp)
        assert p._session_expired is False

    @pytest.mark.asyncio
    async def test_rotation_after_lifetime(self):
        p = EnvProtocolB(session_lifetime=0.1)
        p._session_start = time.monotonic() - 0.2
        assert p.needs_rotation is True

    @pytest.mark.asyncio
    async def test_create_context_with_session(self, tmp_path):
        browser = _make_browser_mock()
        new_ctx = MagicMock()
        new_ctx.add_init_script = AsyncMock()
        browser.new_context = AsyncMock(return_value=new_ctx)

        proxy = _make_proxy()
        fp = {
            "viewport": {"width": 1280, "height": 800},
            "user_agent": "Mozilla/5.0",
            "locale": "en-US",
            "timezone_id": "UTC",
            "device_scale_factor": 1.0,
        }

        ctx = await EnvProtocolB.create_context_with_session(
            browser, proxy, fp, storage_path=None
        )
        assert ctx is new_ctx
        browser.new_context.assert_called_once()
        new_ctx.add_init_script.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_context_with_session_reinjects_cookies(self, tmp_path):
        state = {"cookies": [{"name": "x", "value": "1"}]}
        path = tmp_path / "session.json"
        path.write_text(json.dumps(state))

        browser = _make_browser_mock()
        new_ctx = MagicMock()
        new_ctx.add_init_script = AsyncMock()
        browser.new_context = AsyncMock(return_value=new_ctx)

        proxy = _make_proxy()
        fp = {
            "viewport": {"width": 1280, "height": 800},
            "user_agent": "Mozilla/5.0",
            "locale": "en-US",
            "timezone_id": "UTC",
            "device_scale_factor": 1.0,
        }

        ctx = await EnvProtocolB.create_context_with_session(
            browser, proxy, fp, storage_path=path
        )
        call = browser.new_context.call_args
        assert "storage_state" in call.kwargs
        assert call.kwargs["storage_state"] == state

    @pytest.mark.asyncio
    async def test_teardown_removes_listener(self):
        page = _make_page_mock()
        p = EnvProtocolB()
        await p.prepare(page, _make_context_mock(), _make_proxy(), "#table")
        p.teardown(page)
        page.remove_listener.assert_called_with("response", p._on_response)


# ─────────────────────────────────────────────────────────────────────────────
# Environment C — Deeply nested iframes
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvProtocolC:
    def test_suggested_interval(self):
        p = EnvProtocolC()
        for _ in range(10):
            iv = p.suggested_interval()
            assert 3.0 <= iv <= 8.0

    def test_build_robust_selectors(self):
        p = EnvProtocolC()
        sels = p._build_robust_selectors("my-table")
        assert sels[0] == "#my-table"
        assert "table" in sels
        assert any("data-testid" in s for s in sels)

    @pytest.mark.asyncio
    async def test_rediscover_no_page(self):
        p = EnvProtocolC()
        # Should not raise
        await p._rediscover()
        assert p._stale is True

    @pytest.mark.asyncio
    async def test_rediscover_finds_table(self):
        page = _make_page_mock()
        mock_frame = MagicMock()
        mock_frame.query_selector = AsyncMock(return_value="found")
        page.frames = [mock_frame]

        p = EnvProtocolC()
        p._page = page
        p._table_selector_hint = "#my-table"
        await p._rediscover()

        assert p._stale is False
        assert p._table_frame == mock_frame
        assert p._resolved_selector == "#my-table"

    @pytest.mark.asyncio
    async def test_rediscover_not_found(self):
        page = _make_page_mock()
        mock_frame = MagicMock()
        mock_frame.query_selector = AsyncMock(return_value=None)
        mock_frame.child_frames = []
        page.frames = [mock_frame]

        p = EnvProtocolC()
        p._page = page
        p._table_selector_hint = "#nonexistent"
        await p._rediscover()

        assert p._stale is True
        assert p._table_frame is None

    @pytest.mark.asyncio
    async def test_scrape_table_from_frame(self):
        p = EnvProtocolC()
        p._resolved_selector = "#my-table"

        mock_tr = MagicMock()
        mock_td_ts = MagicMock()
        mock_td_ev = MagicMock()
        mock_td_num = MagicMock()
        mock_td_ts.inner_text = AsyncMock(return_value="2026-01-01")
        mock_td_ev.inner_text = AsyncMock(return_value="CPU_SPIKE")
        mock_td_num.inner_text = AsyncMock(return_value="87.5")

        # query_selector_all returns 3 tds
        mock_tr.query_selector_all = AsyncMock(return_value=[
            mock_td_ts, mock_td_ev, mock_td_num
        ])

        frame = MagicMock()
        frame.query_selector_all = AsyncMock(return_value=[mock_tr])

        rows = await p.scrape_table(MagicMock(), frame, "#my-table", {
            "timestamp": 0, "event_name": 1, "numeric_result": 2
        })
        assert rows == [DataRow("2026-01-01", "CPU_SPIKE", "87.5")]

    @pytest.mark.asyncio
    async def test_scrape_table_blank_row_skipped(self):
        p = EnvProtocolC()
        p._resolved_selector = "#my-table"

        mock_tr = MagicMock()
        mock_td_ts = MagicMock()
        mock_td_ev = MagicMock()
        mock_td_ts.inner_text = AsyncMock(return_value="")
        mock_td_ev.inner_text = AsyncMock(return_value="")
        mock_tr.query_selector_all = AsyncMock(return_value=[
            mock_td_ts, mock_td_ev
        ])

        frame = MagicMock()
        frame.query_selector_all = AsyncMock(return_value=[mock_tr])

        rows = await p.scrape_table(MagicMock(), frame, "#my-table", {
            "timestamp": 0, "event_name": 1, "numeric_result": 2
        })
        assert rows == []

    @pytest.mark.asyncio
    async def test_discover_columns_from_frame(self):
        p = EnvProtocolC()
        mock_th_ts = MagicMock()
        mock_th_ts.inner_text = AsyncMock(return_value="Timestamp")
        mock_th_ev = MagicMock()
        mock_th_ev.inner_text = AsyncMock(return_value="Event Name")
        mock_th_num = MagicMock()
        mock_th_num.inner_text = AsyncMock(return_value="Numeric Result")
        frame = MagicMock()
        frame.query_selector_all = AsyncMock(return_value=[
            mock_th_ts, mock_th_ev, mock_th_num
        ])

        result = await p.discover_columns(frame, "#my-table")
        assert result == {"timestamp": 0, "event_name": 1, "numeric_result": 2}

    @pytest.mark.asyncio
    async def test_find_table_by_headers(self):
        p = EnvProtocolC()
        frame = MagicMock()
        frame.evaluate = AsyncMock(return_value=2)  # 2nd table

        result = await p._find_table_by_headers(frame)
        assert result == "table:nth-of-type(2)"

    @pytest.mark.asyncio
    async def test_find_table_by_headers_no_match(self):
        p = EnvProtocolC()
        frame = MagicMock()
        frame.evaluate = AsyncMock(return_value=None)

        result = await p._find_table_by_headers(frame)
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_data_stale_rediscovery(self):
        p = EnvProtocolC()
        p._stale = True
        p._page = MagicMock()
        # _rediscover will fail to find, leaving _stale True
        await p._rediscover()

        result = await p.wait_for_data(MagicMock(), "#table", time.monotonic(), timeout_ms=100)
        assert result is False

    @pytest.mark.asyncio
    async def test_teardown_removes_listeners(self):
        page = _make_page_mock()
        p = EnvProtocolC()
        await p.prepare(page, MagicMock(), _make_proxy(), "#table")
        p.teardown(page)
        page.remove_listener.assert_any_call("frameattached", p._on_frame_event)
        page.remove_listener.assert_any_call("framenavigated", p._on_frame_event)

    @pytest.mark.asyncio
    async def test_prepare_registers_listeners(self):
        page = _make_page_mock()
        p = EnvProtocolC()
        await p.prepare(page, MagicMock(), _make_proxy(), "#table")
        page.on.assert_any_call("frameattached", p._on_frame_event)
        page.on.assert_any_call("framenavigated", p._on_frame_event)