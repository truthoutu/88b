"""
execution_engine.py
===================
Phase 3: Automated Response Integration for the realtime-dashboard-scraper.

Consumes PredictabilityAnalyzer output and converts high-confidence signals
into protocol-aware browser automation via the existing EnvProtocolA/B/C.

Architecture
------------
    SignalCondition      -- WHEN to trigger (predictability + transition matrix)
    TransitionMatrixMatch -- Transition-matrix pattern criteria
    CircuitBreaker        -- Risk management (halts execution on anomalies)
    Payload               -- HOW to interact (click/type/select with jitter)
    Action                -- Modular action definition (locate + deliver + confirm)
    ExecutionContext      -- Protocol-aware wrapper around Page/Frame/Protocol
    ExecutionEngine       -- Main orchestrator
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger
from playwright.async_api import (
    BrowserContext,
    ElementHandle,
    Frame,
    Page,
    TimeoutError as PWTimeoutError,
)

from env_optimizations import _BaseProtocol
from predictability_analyzer import PredictabilityResult


# ─────────────────────────────────────────────────────────────────────────────
# Signal integration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitionMatrixMatch:
    """
    Criteria for matching a specific transition-matrix pattern.

    Attributes
    ----------
    strongest_state : int | None
        If set, the dominant state index must equal this value.
    min_transition_score : float
        Minimum transition matrix score (from PredictabilityResult).
    min_normalized_mi : float
        Minimum normalized mutual information (from PredictabilityResult).
    custom_predicate : Callable[[np.ndarray, int, int], bool] | None
        Optional user-defined function: (matrix, n_states, strongest_state) -> bool.
    """
    strongest_state: int | None = None
    min_transition_score: float = 0.0
    min_normalized_mi: float = 0.0
    custom_predicate: Callable[..., bool] | None = None

    def matches(self, result: PredictabilityResult) -> bool:
        """Return True if the transition matrix satisfies all criteria."""
        if self.strongest_state is not None:
            if result.autocorrelation.dominant_lag != self.strongest_state:
                return False

        if result.autocorrelation.periodicity_score < self.min_transition_score:
            return False

        if result.conditional.normalized_mi < self.min_normalized_mi:
            return False

        if self.custom_predicate is not None:
            try:
                if not self.custom_predicate(
                    result.conditional.transition_matrix,
                    result.conditional.n_states,
                    result.autocorrelation.dominant_lag,
                ):
                    return False
            except Exception as exc:
                logger.warning("[Exec] custom_predicate raised: {}", exc)
                return False

        return True


@dataclass
class SignalCondition:
    """
    Defines WHEN the ExecutionEngine should trigger actions.

    Attributes
    ----------
    predictability_threshold : float
        Minimum predictability score in [0, 1].
    transition_match : TransitionMatrixMatch | None
        Optional transition-matrix criteria.
    """
    predictability_threshold: float = 0.75
    transition_match: TransitionMatrixMatch | None = None

    def matches(self, result: PredictabilityResult) -> bool:
        """Return True if the signal satisfies all conditions."""
        if result.predictability_score < self.predictability_threshold:
            return False

        if result.n_samples < 20:
            logger.debug("[Exec] Insufficient samples ({}) for reliable signal", result.n_samples)
            return False

        if self.transition_match is not None:
            return self.transition_match.matches(result)

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Risk management
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CircuitBreaker:
    """
    Halts all execution when risk thresholds are breached.

    Trip conditions
    ---------------
    1. Predictability score drops below threshold on any evaluation.
    2. Unusual increase in 403 Forbidden errors within a rolling window.
    3. Sequence of three consecutive failed confirmation loops.
    """

    predictability_threshold: float = 0.75
    max_consecutive_failures: int = 3
    forbidden_error_window_s: float = 60.0
    max_forbidden_in_window: int = 5
    auto_reset_after_s: float = 300.0

    is_open: bool = False
    trip_reason: str | None = None
    consecutive_failures: int = 0
    forbidden_timestamps: list[float] = field(default_factory=list)
    _last_reset: float = field(default_factory=time.monotonic)

    def can_execute(self, predictability_score: float | None = None) -> bool:
        """Return True if execution is allowed; auto-resets after cooldown."""
        if self.is_open:
            if time.monotonic() - self._last_reset > self.auto_reset_after_s:
                self.reset()
                logger.info("[CircuitBreaker] Auto-reset after {}s", self.auto_reset_after_s)
            else:
                return False

        if predictability_score is not None and predictability_score < self.predictability_threshold:
            self._trip(
                "predictability_drop (score={:.3f} < {})".format(
                    predictability_score, self.predictability_threshold
                )
            )
            return False

        self._prune_forbidden()
        if len(self.forbidden_timestamps) > self.max_forbidden_in_window:
            self._trip(
                "forbidden_spike ({} errors in {}s window)".format(
                    len(self.forbidden_timestamps), self.forbidden_error_window_s
                )
            )
            return False

        return True

    def record_success(self) -> None:
        """Reset consecutive-failure counter on a successful confirmation."""
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Increment consecutive-failure counter and trip if threshold breached."""
        self.consecutive_failures += 1
        logger.warning(
            "[CircuitBreaker] Consecutive failures: {}/{}",
            self.consecutive_failures,
            self.max_consecutive_failures,
        )
        if self.consecutive_failures >= self.max_consecutive_failures:
            self._trip("consecutive_failures ({})".format(self.consecutive_failures))

    def record_forbidden(self) -> None:
        """Record a 403 Forbidden timestamp."""
        self.forbidden_timestamps.append(time.monotonic())

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self.is_open = False
        self.trip_reason = None
        self.consecutive_failures = 0
        self.forbidden_timestamps.clear()
        self._last_reset = time.monotonic()

    def _trip(self, reason: str) -> None:
        self.is_open = True
        self.trip_reason = reason
        logger.critical("[CircuitBreaker] TRIPPED: {}. All execution halted.", reason)

    def _prune_forbidden(self) -> None:
        cutoff = time.monotonic() - self.forbidden_error_window_s
        self.forbidden_timestamps = [t for t in self.forbidden_timestamps if t > cutoff]


# ─────────────────────────────────────────────────────────────────────────────
# Payload
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Payload:
    """
    Defines how to interact with a located element.

    Attributes
    ----------
    action_type : str
        One of: click, type, select, hover, wait, screenshot.
    value : str | None
        Text for ``type``, option value for ``select``, path for screenshot.
    jitter_ms : tuple[int, int]
        Min/max human-like delay in milliseconds before delivery.
    confirm_timeout_ms : int
        Timeout for the confirmation loop after delivery.
    """

    action_type: str = "click"
    value: str | None = None
    jitter_ms: tuple[int, int] = (50, 250)
    confirm_timeout_ms: int = 5_000


# ─────────────────────────────────────────────────────────────────────────────
# Action (abstract base + concrete subclasses)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionResult:
    """Outcome of a single action execution."""
    action_name: str
    success: bool
    confirmation_passed: bool
    error: str | None = None
    duration_ms: float = 0.0


class Action(ABC):
    """
    Abstract base for all action routines.

    Subclasses must implement:
    * locate_target(ctx) -> ElementHandle | Frame | None
    * confirm(ctx, target) -> bool
    """

    name: str = "unnamed"
    env_requirement: str = "ANY"  # A | B | C | ANY
    payload: Payload = field(default_factory=Payload)

    @abstractmethod
    async def locate_target(self, ctx: "ExecutionContext") -> ElementHandle | Frame | None:
        """Locate the interactive element or frame."""

    @abstractmethod
    async def confirm(self, ctx: "ExecutionContext", target: ElementHandle | Frame | None) -> bool:
        """Verify the post-action DOM state indicates success."""

    async def execute(self, ctx: "ExecutionContext") -> ActionResult:
        """
        Full execution cycle: locate → jitter → deliver → confirm.
        """
        start = time.monotonic()
        logger.info("[Exec] Action '{}' starting on env={}", self.name, ctx.env)

        try:
            target = await self.locate_target(ctx)
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            logger.error("[Exec] Action '{}' locate failed: {}", self.name, exc)
            return ActionResult(self.name, False, False, str(exc), duration)

        if target is None:
            duration = (time.monotonic() - start) * 1000
            logger.error("[Exec] Action '{}' target not found", self.name)
            return ActionResult(self.name, False, False, "target not found", duration)

        # Human-like jitter before delivery
        jitter = random.uniform(*self.payload.jitter_ms) / 1000
        await asyncio.sleep(jitter)

        try:
            delivered = await self._deliver(ctx, target)
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            logger.error("[Exec] Action '{}' delivery failed: {}", self.name, exc)
            return ActionResult(self.name, False, False, f"delivery error: {exc}", duration)

        if not delivered:
            duration = (time.monotonic() - start) * 1000
            return ActionResult(self.name, False, False, "delivery returned False", duration)

        # Confirmation loop
        try:
            confirmed = await asyncio.wait_for(
                self.confirm(ctx, target),
                timeout=self.payload.confirm_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            duration = (time.monotonic() - start) * 1000
            logger.warning("[Exec] Action '{}' confirmation timed out", self.name)
            return ActionResult(self.name, True, False, "confirmation timeout", duration)
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            logger.error("[Exec] Action '{}' confirmation error: {}", self.name, exc)
            return ActionResult(self.name, True, False, f"confirmation error: {exc}", duration)

        duration = (time.monotonic() - start) * 1000
        logger.success(
            "[Exec] Action '{}' completed in {:.0f}ms confirmed={}",
            self.name,
            duration,
            confirmed,
        )
        return ActionResult(self.name, True, confirmed, duration_ms=duration)

    async def _deliver(self, ctx: "ExecutionContext", target: ElementHandle | Frame) -> bool:
        """Perform the actual click/type/select/hover/wait."""
        action = self.payload.action_type.lower()

        if isinstance(target, Frame):
            return False

        try:
            if action == "click":
                await target.click()
                return True

            elif action == "type":
                text = self.payload.value or ""
                await target.fill(text)
                return True

            elif action == "select":
                value = self.payload.value or ""
                await target.select_option(value=value)
                return True

            elif action == "hover":
                await target.hover()
                return True

            elif action == "wait":
                await asyncio.sleep(float(self.payload.value or 1.0))
                return True

            elif action == "screenshot":
                path = self.payload.value or "action_screenshot.png"
                await ctx.page.screenshot(path=path)
                return True

            else:
                logger.error("[Exec] Unknown action_type: {}", action)
                return False
        except PWTimeoutError:
            logger.error("[Exec] Action '{}' delivery timed out", self.name)
            return False
        self._prune_forbidden()
        if len(self.forbidden_timestamps) > self.max_forbidden_in_window:
            self._trip(
                "forbidden_spike ({} errors in {}s window)".format(
                    len(self.forbidden_timestamps), self.forbidden_error_window_s
                )
            )
            return False

        return True

    def record_success(self) -> None:
        """Reset consecutive-failure counter on a successful confirmation."""
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Increment consecutive-failure counter and trip if threshold breached."""
        self.consecutive_failures += 1
        logger.warning(
            "[CircuitBreaker] Consecutive failures: {}/{}",
            self.consecutive_failures,
            self.max_consecutive_failures,
        )
        if self.consecutive_failures >= self.max_consecutive_failures:
            self._trip(f"consecutive_failures ({self.consecutive_failures})")

    def record_forbidden(self) -> None:
        """Record a 403 Forbidden timestamp."""
        self.forbidden_timestamps.append(time.monotonic())

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self.is_open = False
        self.trip_reason = None
        self.consecutive_failures = 0
        self.forbidden_timestamps.clear()
        self._last_reset = time.monotonic()

    # -- internal helpers --

    def _trip(self, reason: str) -> None:
        self.is_open = True
        self.trip_reason = reason
        logger.critical("[CircuitBreaker] TRIPPED: {}. All execution halted.", reason)

    def _prune_forbidden(self) -> None:
        cutoff = time.monotonic() - self.forbidden_error_window_s
        self.forbidden_timestamps = [t for t in self.forbidden_timestamps if t > cutoff]
        return True

class ClickAction(Action):
    """Click a button or interactive element by CSS selector."""

    def __init__(self, selector: str, payload: Payload | None = None):
        self.name = f"click:{selector}"
        self.env_requirement = "ANY"
        self.selector = selector
        self.payload = payload or Payload(action_type="click")

    async def locate_target(self, ctx: "ExecutionContext") -> ElementHandle | None:
        return await ctx.active_frame.query_selector(self.selector)

    async def confirm(self, ctx: "ExecutionContext", target: ElementHandle | None) -> bool:
        if target is None:
            return False
        try:
            is_visible = await target.is_visible()
            return not is_visible
        except Exception:
            return False


class TypeAction(Action):
    """Type text into an input field by CSS selector."""

    def __init__(self, selector: str, text: str, payload: Payload | None = None):
        self.name = f"type:{selector}"
        self.env_requirement = "ANY"
        self.selector = selector
        self.text = text
        self.payload = payload or Payload(action_type="type", value=text)

    async def locate_target(self, ctx: "ExecutionContext") -> ElementHandle | None:
        return await ctx.active_frame.query_selector(self.selector)

    async def confirm(self, ctx: "ExecutionContext", target: ElementHandle | None) -> bool:
        if target is None:
            return False
        try:
            current = await target.input_value()
            return self.text in current
        except Exception:
            return False


class SelectAction(Action):
    """Select an option from a dropdown by CSS selector."""

    def __init__(self, selector: str, option_value: str, payload: Payload | None = None):
        self.name = f"select:{selector}"
        self.env_requirement = "ANY"
        self.selector = selector
        self.option_value = option_value
        self.payload = payload or Payload(action_type="select", value=option_value)

    async def locate_target(self, ctx: "ExecutionContext") -> ElementHandle | None:
        return await ctx.active_frame.query_selector(self.selector)

    async def confirm(self, ctx: "ExecutionContext", target: ElementHandle | None) -> bool:
        if target is None:
            return False
        try:
            selected = await target.evaluate("el => el.value")
            return selected == self.option_value
        except Exception:
            return False

class CoordinateClickAction(Action):
    """Click at dynamic coordinates with optional jitter offset."""

    def __init__(
        self,
        x: int,
        y: int,
        jitter_px: int = 5,
        payload: Payload | None = None,
    ):
        self.name = f"coord:{x},{y}"
        self.env_requirement = "ANY"
        self.x = x
        self.y = y
        self.jitter_px = jitter_px
        self.payload = payload or Payload(action_type="click")

    async def locate_target(self, ctx: "ExecutionContext") -> ElementHandle | None:
        return None

    async def confirm(self, ctx: "ExecutionContext", target: ElementHandle | None) -> bool:
        return True

    async def execute(self, ctx: "ExecutionContext") -> ActionResult:
        """Skip target lookup; deliver directly via mouse click."""
        start = time.monotonic()
        logger.info("[Exec] Action '{}' starting on env={}", self.name, ctx.env)

        jitter = random.uniform(*self.payload.jitter_ms) / 1000
        await asyncio.sleep(jitter)

        try:
            delivered = await self._deliver(ctx, None)
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            logger.error("[Exec] Action '{}' delivery failed: {}", self.name, exc)
            return ActionResult(self.name, False, False, f"delivery error: {exc}", duration)

        if not delivered:
            duration = (time.monotonic() - start) * 1000
            return ActionResult(self.name, False, False, "delivery returned False", duration)

        duration = (time.monotonic() - start) * 1000
        logger.success("[Exec] Action '{}' completed in {:.0f}ms", self.name, duration)
        return ActionResult(self.name, True, True, duration_ms=duration)

    async def _deliver(self, ctx: "ExecutionContext", target: ElementHandle | Frame) -> bool:
        jx = random.randint(-self.jitter_px, self.jitter_px)
        jy = random.randint(-self.jitter_px, self.jitter_px)
        await ctx.page.mouse.click(self.x + jx, self.y + jy)
        return True


class FrameClickAction(Action):
    """EnvC-specific: resolve an iframe and click a button inside it."""

    def __init__(self, iframe_hint: str, button_selector: str, payload: Payload | None = None):
        self.name = f"frame_click:{iframe_hint}->{button_selector}"
        self.env_requirement = "C"
        self.iframe_hint = iframe_hint
        self.button_selector = button_selector
        self.payload = payload or Payload(action_type="click")

    async def locate_target(self, ctx: "ExecutionContext") -> ElementHandle | None:
        protocol = ctx.protocol
        frame = getattr(protocol, "table_frame", None)
        if frame is None or not isinstance(frame, Frame):
            try:
                frame, _ = await protocol.find_table_frame(ctx.page, self.iframe_hint)
            except Exception:
                return None

        try:
            return await frame.query_selector(self.button_selector)
        except Exception:
            return None

    async def confirm(self, ctx: "ExecutionContext", target: ElementHandle | None) -> bool:
        if target is None:
            return False
        try:
            is_visible = await target.is_visible()
            return not is_visible
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionContext
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionContext:
    """
    Protocol-aware wrapper around Page / Frame / Protocol.

    Attributes
    ----------
    env : str
        "A" | "B" | "C"
    protocol : _BaseProtocol
        The environment-specific protocol instance.
    page : Page
        The Playwright page.
    context : BrowserContext
        The browser context (relevant for EnvB session management).
    frame : Frame | None
        Override frame; if None, falls back to protocol.table_frame or page.main_frame.
    proxy_cfg : Any
        Proxy configuration dict.
    """

    env: str
    protocol: _BaseProtocol
    page: Page
    context: BrowserContext
    frame: Frame | None = None
    proxy_cfg: Any = None

    @property
    def active_frame(self) -> Frame:
        """Return the frame that action queries should target."""
        if self.frame is not None:
            return self.frame
        table_frame = getattr(self.protocol, "table_frame", None)
        if table_frame is not None:
            return table_frame
        return self.page.main_frame


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionEngine
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionEngine:
    """
    Main orchestrator that evaluates predictability signals and executes
    protocol-aware action routines with risk management.
    """

    def __init__(
        self,
        ctx: ExecutionContext,
        signal_condition: SignalCondition,
        circuit_breaker: CircuitBreaker,
        actions: list[Action] | None = None,
    ) -> None:
        self.ctx = ctx
        self.signal_condition = signal_condition
        self.circuit_breaker = circuit_breaker
        self.actions: list[Action] = list(actions or [])
        self._response_listener_attached = False

    def add_action(self, action: Action) -> None:
        """Register a new action routine."""
        self.actions.append(action)

    async def evaluate_and_execute(
        self, result: PredictabilityResult
    ) -> list[ActionResult]:
        """
        Evaluate a PredictabilityResult and trigger actions if conditions are met.

        Returns
        -------
        list[ActionResult]
            One result per action executed (or empty if no actions triggered).
        """
        action_results: list[ActionResult] = []

        # ── 1. Circuit breaker pre-check ───────────────────────────────
        if not self.circuit_breaker.can_execute(result.predictability_score):
            logger.warning(
                "[Exec] Circuit breaker OPEN ({}). Skipping execution.",
                self.circuit_breaker.trip_reason,
            )
            return action_results

        # ── 2. Signal condition check ──────────────────────────────────
        if not self.signal_condition.matches(result):
            logger.debug(
                "[Exec] Signal condition not met (score={:.3f})",
                result.predictability_score,
            )
            return action_results

        logger.info(
            "[Exec] Signal MATCHED (score={:.3f}). Executing {} action(s).",
            result.predictability_score,
            len(self.actions),
        )

        # ── 3. Attach 403 listener if not already attached ─────────────
        if not self._response_listener_attached:
            self.ctx.page.on("response", self._on_response)
            self._response_listener_attached = True

        # ── 4. Execute actions sequentially ────────────────────────────
        for action in self.actions:
            # Re-check circuit breaker before each action
            if not self.circuit_breaker.can_execute():
                logger.warning("[Exec] Circuit breaker tripped during execution. Aborting.")
                break

            # Skip actions not applicable to current environment
            if action.env_requirement != "ANY" and action.env_requirement != self.ctx.env:
                logger.debug(
                    "[Exec] Skipping '{}' (requires env={})",
                    action.name,
                    action.env_requirement,
                )
                continue

            # Protocol-aware pre-execution hooks
            await _ensure_protocol_ready(self.ctx, action)

            # Execute the action
            result_action = await action.execute(self.ctx)
            action_results.append(result_action)

            # Update circuit breaker state
            if result_action.confirmation_passed:
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_failure()

        return action_results

    async def _on_response(self, response) -> None:
        """Playwright response listener for 403 detection."""
        try:
            if response.status == 403:
                self.circuit_breaker.record_forbidden()
                logger.warning("[Exec] 403 detected on {}", response.url)
        except Exception:
            pass
            return False

# ─────────────────────────────────────────────────────────────────────────────
# Protocol-specific helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _ensure_protocol_ready(ctx: ExecutionContext, action: Action) -> None:
    """
    Protocol-aware pre-execution hooks.

    * EnvA: ensure WS is synchronized before interacting.
    * EnvB: re-inject cookies after context rotation.
    * EnvC: re-discover table frame if stale.
    """
    protocol = ctx.protocol
    env = ctx.env

    if env == "A":
        table_sel = getattr(protocol, "_table_sel", None)
        if table_sel:
            try:
                await protocol.wait_for_data(ctx.page, table_sel, None)
            except Exception:
                pass

    elif env == "B":
        storage_path = getattr(protocol, "_storage_path", None)
        if storage_path:
            try:
                import json
                import os

                if os.path.exists(storage_path):
                    with open(storage_path) as fh:
                        state = json.load(fh)
                    cookies = state.get("cookies", [])
                    if cookies:
                        await ctx.context.add_cookies(cookies)
            except Exception:
                pass

    elif env == "C":
        if getattr(protocol, "_stale", False):
            try:
                await protocol._rediscover(ctx.page)
            except Exception:
                pass