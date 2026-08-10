"""
protocol_executors.py
=====================
Small helper functions that adapt the generic Action interface to each
environment's quirks. Kept separate from execution_engine.py to maintain
a clean separation between environment-agnostic orchestration and
environment-specific browser protocol details.

Available helpers
-----------------
* ensure_ws_synchronized(protocol, page, table_sel)
* ensure_valid_session(protocol, context, storage_path)
* resolve_action_frame(protocol, page, iframe_hint)
"""

from __future__ import annotations

import json
import os

from loguru import logger

from env_optimizations import _BaseProtocol


async def ensure_ws_synchronized(protocol: _BaseProtocol, page, table_sel: str) -> None:
    """
    EnvA helper: block until the WebSocket has pushed at least one data frame.

    Uses EnvProtocolA.wait_for_data() which polls the tbody row count until it
    increases, ensuring the table reflects the latest WS-driven update before
    any action is delivered.
    """
    try:
        await protocol.wait_for_data(page, table_sel, None)
    except Exception as exc:
        logger.warning("[ProtocolExecutor] WS sync failed: {}", exc)


async def ensure_valid_session(
    protocol: _BaseProtocol,
    context,
    storage_path: str,
) -> None:
    """
    EnvB helper: re-inject saved cookies into a freshly created context.

    Should be called immediately after ``create_context_with_session()`` and
    also periodically after successful actions to keep the session consistent.
    """
    if not storage_path or not os.path.exists(storage_path):
        return

    try:
        with open(storage_path) as fh:
            state = json.load(fh)
        cookies = state.get("cookies", [])
        if cookies:
            await context.add_cookies(cookies)
            logger.debug("[ProtocolExecutor] Re-injected {} cookies", len(cookies))
    except Exception as exc:
        logger.warning("[ProtocolExecutor] Session rehydration failed: {}", exc)


async def resolve_action_frame(
    protocol: _BaseProtocol,
    page,
    iframe_hint: str,
) -> tuple[object, str | None]:
    """
    EnvC helper: resolve the iframe containing the interactive target.

    Returns
    -------
    (frame, None) if found, (None, error_message) otherwise.
    """
    try:
        frame, _ = await protocol.find_table_frame(page, iframe_hint)
        if frame is None:
            return None, "frame not found"
        return frame, None
    except Exception as exc:
        return None, str(exc)