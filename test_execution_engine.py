"""
test_execution_engine.py
========================
Unit tests for execution_engine.py covering:

* SignalCondition + TransitionMatrixMatch
* CircuitBreaker (trip/auto-reset/forbidden window)
* Payload
* Action subclasses (ClickAction, TypeAction, SelectAction, CoordinateClickAction, FrameClickAction)
* ExecutionContext.active_frame fallback
* ExecutionEngine.evaluate_and_execute (signal match, breaker trip, 403 listener)
* protocol_executors helpers
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from env_optimizations import EnvProtocolA, EnvProtocolC
from execution_engine import (
    CircuitBreaker,
    ClickAction,
    CoordinateClickAction,
    ExecutionContext,
    ExecutionEngine,
    FrameClickAction,
    Payload,
    SelectAction,
    SignalCondition,
    TransitionMatrixMatch,
    TypeAction,
)
from predictability_analyzer import (
    AutocorrelationResult,
    ChangePointResult,
    ConditionalProbabilityResult,
    FrequencyAnalysis,
    PredictabilityResult,
)
from playwright.async_api import Frame


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_result_high() -> PredictabilityResult:
    """PredictabilityResult with high predictability."""
    return PredictabilityResult(
        n_samples=100,
        weights={"frequency": 0.25, "autocorrelation": 0.35, "conditional": 0.25, "change_point": 0.15},
        frequency=FrequencyAnalysis(
            observed_counts=np.array([10, 10, 10]),
            expected_per_bin=10.0,
            chi2_stat=0.0,
            chi2_pvalue=1.0,
            cohen_w=0.0,
            deviation_score=0.0,
            ks_statistic=0.0,
            ks_pvalue=1.0,
            is_uniform=True,
        ),
        autocorrelation=AutocorrelationResult(
            acf=np.array([1.0, 0.5, 0.82]),
            confidence_band=0.2,
            significant_lags=[2],
            dominant_lag=2,
            dominant_acf=0.82,
            periodicity_score=0.82,
        ),
        conditional=ConditionalProbabilityResult(
            n_states=2,
            transition_matrix=np.array([[0.5, 0.5], [0.3, 0.7]]),
            mutual_information=0.5,
            normalized_mi=0.7,
            transition_score=0.6,
            strongest_state=1,
        ),
        change_point=ChangePointResult(
            change_points=[],
            max_ks_statistic=0.0,
            n_significant=0,
            change_score=0.0,
        ),
        sub_scores={"frequency": 0.9, "autocorrelation": 0.85, "conditional": 0.8, "change_point": 0.9},
        predictability_score=0.85,
        interpretation="highly predictable",
        notes=[],
    )


@pytest.fixture()
def sample_result_low() -> PredictabilityResult:
    """PredictabilityResult with low predictability."""
    return PredictabilityResult(
        n_samples=100,
        weights={"frequency": 0.25, "autocorrelation": 0.35, "conditional": 0.25, "change_point": 0.15},
        frequency=FrequencyAnalysis(
            observed_counts=np.array([5, 5, 5]),
            expected_per_bin=5.0,
            chi2_stat=0.0,
            chi2_pvalue=1.0,
            cohen_w=0.0,
            deviation_score=0.0,
            ks_statistic=0.0,
            ks_pvalue=1.0,
            is_uniform=True,
        ),
        autocorrelation=AutocorrelationResult(
            acf=np.array([1.0, 0.1]),
            confidence_band=0.2,
            significant_lags=[],
            dominant_lag=0,
            dominant_acf=0.1,
            periodicity_score=0.1,
        ),
        conditional=ConditionalProbabilityResult(
            n_states=2,
            transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
            mutual_information=0.0,
            normalized_mi=0.1,
            transition_score=0.0,
            strongest_state=0,
        ),
        change_point=ChangePointResult(
            change_points=[],
            max_ks_statistic=0.0,
            n_significant=0,
            change_score=0.0,
        ),
        sub_scores={"frequency": 0.3, "autocorrelation": 0.2, "conditional": 0.3, "change_point": 0.4},
        predictability_score=0.30,
        interpretation="low predictability",
        notes=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: TransitionMatrixMatch
# ─────────────────────────────────────────────────────────────────────────────

class TestTransitionMatrixMatch:
    def test_matches_strongest_state(self, sample_result_high):
        tm = TransitionMatrixMatch(strongest_state=2, min_transition_score=0.5)
        assert tm.matches(sample_result_high) is True

    def test_matches_strongest_state_fails(self, sample_result_high):
        tm = TransitionMatrixMatch(strongest_state=99, min_transition_score=0.0)
        assert tm.matches(sample_result_high) is False

    def test_matches_transition_score_threshold(self, sample_result_high):
        tm = TransitionMatrixMatch(strongest_state=2, min_transition_score=0.9)
        assert tm.matches(sample_result_high) is False

    def test_matches_normalized_mi_threshold(self, sample_result_high):
        tm = TransitionMatrixMatch(min_normalized_mi=0.9)
        assert tm.matches(sample_result_high) is False

    def test_custom_predicate_true(self, sample_result_high):
        tm = TransitionMatrixMatch(
            custom_predicate=lambda m, n, s: n == 2,
        )
        assert tm.matches(sample_result_high) is True

    def test_custom_predicate_false(self, sample_result_high):
        tm = TransitionMatrixMatch(
            custom_predicate=lambda m, n, s: n == 99,
        )
        assert tm.matches(sample_result_high) is False

    def test_custom_predicate_raises_returns_false(self, sample_result_high):
        tm = TransitionMatrixMatch(
            custom_predicate=lambda m, n, s: (_ for _ in ()).throw(ValueError("boom")),
        )
        assert tm.matches(sample_result_high) is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests: SignalCondition
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalCondition:
    def test_matches_high_score_no_transition(self, sample_result_high):
        sc = SignalCondition(predictability_threshold=0.75)
        assert sc.matches(sample_result_high) is True

    def test_matches_low_score(self, sample_result_low):
        sc = SignalCondition(predictability_threshold=0.75)
        assert sc.matches(sample_result_low) is False

    def test_matches_insufficient_samples(self, sample_result_high):
        sc = SignalCondition(predictability_threshold=0.75)
        sample_result_high.n_samples = 5
        assert sc.matches(sample_result_high) is False

    def test_matches_with_transition_matrix(self, sample_result_high):
        sc = SignalCondition(
            predictability_threshold=0.75,
            transition_match=TransitionMatrixMatch(strongest_state=2, min_transition_score=0.5),
        )
        assert sc.matches(sample_result_high) is True

    def test_matches_with_transition_matrix_fails(self, sample_result_high):
        sc = SignalCondition(
            predictability_threshold=0.75,
            transition_match=TransitionMatrixMatch(strongest_state=2, min_transition_score=0.9),
        )
        assert sc.matches(sample_result_high) is False

# ─────────────────────────────────────────────────────────────────────────────
# Tests: CircuitBreaker
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_can_execute_initially_true(self):
        cb = CircuitBreaker()
        assert cb.can_execute(0.9) is True

    def test_trips_on_low_predictability(self):
        cb = CircuitBreaker(predictability_threshold=0.75)
        assert cb.can_execute(0.5) is False
        assert cb.is_open is True
        assert "predictability_drop" in cb.trip_reason

    def test_trips_on_consecutive_failures(self):
        cb = CircuitBreaker(max_consecutive_failures=3)
        assert cb.can_execute(0.9) is True
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute(0.9) is True
        cb.record_failure()
        assert cb.is_open is True
        assert "consecutive_failures" in cb.trip_reason

    def test_success_resets_failures(self):
        cb = CircuitBreaker(max_consecutive_failures=2)
        cb.record_failure()
        assert cb.consecutive_failures == 1
        cb.record_success()
        assert cb.consecutive_failures == 0

    def test_trips_on_forbidden_spike(self):
        cb = CircuitBreaker(max_forbidden_in_window=3, forbidden_error_window_s=60.0)
        now = time.monotonic()
        with patch("time.monotonic", return_value=now):
            cb.record_forbidden()
            cb.record_forbidden()
            cb.record_forbidden()
            cb.record_forbidden()
            assert cb.can_execute(0.9) is False
            assert "forbidden_spike" in cb.trip_reason

    def test_prune_old_forbidden(self):
        cb = CircuitBreaker(max_forbidden_in_window=3, forbidden_error_window_s=10.0)
        old = time.monotonic() - 20.0
        new = time.monotonic()
        with patch("time.monotonic", side_effect=[old, old, old, old, new]):
            cb.record_forbidden()
            cb.record_forbidden()
            cb.record_forbidden()
        with patch("time.monotonic", return_value=new):
            assert cb.can_execute(0.9) is True
            assert len(cb.forbidden_timestamps) == 0

    def test_reset_clears_state(self):
        cb = CircuitBreaker()
        cb._trip("test")
        cb.reset()
        assert cb.is_open is False
        assert cb.trip_reason is None
        assert cb.consecutive_failures == 0

    def test_auto_reset_after_cooldown(self):
        cb = CircuitBreaker(auto_reset_after_s=0.5)
        cb._trip("test")
        time.sleep(0.6)
        assert cb.can_execute(0.9) is True
        assert cb.is_open is False

# ─────────────────────────────────────────────────────────────────────────────
# Tests: Action subclasses
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_ctx():
    """Minimal ExecutionContext with mocked protocol/page/context/frame."""
    protocol = MagicMock(spec=EnvProtocolA)
    page = MagicMock()
    context = MagicMock()
    frame = MagicMock()
    ctx = ExecutionContext(
        env="A",
        protocol=protocol,
        page=page,
        context=context,
        frame=frame,
    )
    return ctx, protocol, page, context, frame


class TestClickAction:
    @pytest.mark.asyncio
    async def test_locate_returns_element(self, mock_ctx):
        ctx, _, _, _, frame = mock_ctx
        element = MagicMock()
        frame.query_selector = AsyncMock(return_value=element)
        action = ClickAction("#btn")
        target = await action.locate_target(ctx)
        frame.query_selector.assert_awaited_once_with("#btn")
        assert target is element

    @pytest.mark.asyncio
    async def test_confirm_when_not_visible(self, mock_ctx):
        ctx, _, _, _, frame = mock_ctx
        element = MagicMock()
        element.is_visible = AsyncMock(return_value=False)
        action = ClickAction("#btn")
        confirmed = await action.confirm(ctx, element)
        assert confirmed is True

    @pytest.mark.asyncio
    async def test_full_execute(self, mock_ctx):
        ctx, _, page, _, frame = mock_ctx
        element = MagicMock()
        element.is_visible = AsyncMock(return_value=False)
        frame.query_selector = AsyncMock(return_value=element)
        element.click = AsyncMock()
        action = ClickAction("#btn", Payload(jitter_ms=(0, 0), confirm_timeout_ms=100))
        result = await action.execute(ctx)
        assert result.success is True
        assert result.confirmation_passed is True
        element.click.assert_awaited_once()


class TestTypeAction:
    @pytest.mark.asyncio
    async def test_full_execute(self, mock_ctx):
        ctx, _, _, _, frame = mock_ctx
        element = MagicMock()
        element.input_value = AsyncMock(return_value="hello world")
        frame.query_selector = AsyncMock(return_value=element)

        async def fake_fill(text):
            pass

        element.fill = fake_fill
        action = TypeAction("#input", "hello", payload=Payload(action_type="type", value="hello", jitter_ms=(0, 0), confirm_timeout_ms=100))
        result = await action.execute(ctx)
        assert result.success is True
        assert result.confirmation_passed is True


class TestSelectAction:
    @pytest.mark.asyncio
    async def test_full_execute(self, mock_ctx):
        ctx, _, _, _, frame = mock_ctx
        element = MagicMock()
        element.evaluate = AsyncMock(return_value="opt2")
        frame.query_selector = AsyncMock(return_value=element)

        async def fake_select(**kwargs):
            pass

        element.select_option = fake_select
        action = SelectAction("#sel", "opt2", payload=Payload(action_type="select", value="opt2", jitter_ms=(0, 0), confirm_timeout_ms=100))
        result = await action.execute(ctx)
        assert result.success is True
        assert result.confirmation_passed is True


class TestCoordinateClickAction:
    @pytest.mark.asyncio
    async def test_deliver_applies_jitter(self, mock_ctx):
        ctx, _, page, _, _ = mock_ctx
        page.mouse.click = AsyncMock()
        action = CoordinateClickAction(100, 200, jitter_px=0, payload=Payload(jitter_ms=(0, 0)))
        result = await action.execute(ctx)
        assert result.success is True
        page.mouse.click.assert_awaited_once_with(100, 200)


class TestFrameClickAction:
    @pytest.mark.asyncio
    async def test_resolves_frame_and_click(self):
        protocol = MagicMock(spec=EnvProtocolC)
        page = MagicMock()
        context = MagicMock()
        frame = MagicMock(spec=Frame)
        button = MagicMock()
        button.is_visible = AsyncMock(return_value=False)

        async def fake_click():
            pass

        button.click = fake_click
        frame.query_selector = AsyncMock(return_value=button)
        protocol.find_table_frame = AsyncMock(return_value=(frame, None))
        protocol.table_frame = None

        ctx = ExecutionContext(env="C", protocol=protocol, page=page, context=context)
        action = FrameClickAction("iframe-hint", "button.sel", Payload(jitter_ms=(0, 0), confirm_timeout_ms=100))
        result = await action.execute(ctx)
        assert result.success is True
        assert result.confirmation_passed is True

# ─────────────────────────────────────────────────────────────────────────────
# Tests: ExecutionContext
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionContext:
    def test_active_frame_override(self):
        override = MagicMock()
        ctx = ExecutionContext(
            env="A",
            protocol=MagicMock(),
            page=MagicMock(),
            context=MagicMock(),
            frame=override,
        )
        assert ctx.active_frame is override

    def test_active_frame_falls_back_to_table_frame(self):
        table_frame = MagicMock()
        protocol = MagicMock()
        protocol.table_frame = table_frame
        ctx = ExecutionContext(
            env="C",
            protocol=protocol,
            page=MagicMock(),
            context=MagicMock(),
            frame=None,
        )
        assert ctx.active_frame is table_frame

    def test_active_frame_falls_back_to_main_frame(self):
        main_frame = MagicMock()
        page = MagicMock()
        page.main_frame = main_frame
        protocol = MagicMock()
        protocol.table_frame = None
        ctx = ExecutionContext(
            env="A",
            protocol=protocol,
            page=page,
            context=MagicMock(),
            frame=None,
        )
        assert ctx.active_frame is main_frame


# ─────────────────────────────────────────────────────────────────────────────
# Tests: ExecutionEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionEngine:
    @pytest.fixture()
    def engine_setup(self, sample_result_high):
        protocol = MagicMock(spec=EnvProtocolA)
        page = MagicMock()
        page.on = MagicMock()
        context = MagicMock()
        ctx = ExecutionContext(env="A", protocol=protocol, page=page, context=context)

        engine = ExecutionEngine(
            ctx=ctx,
            signal_condition=SignalCondition(predictability_threshold=0.75),
            circuit_breaker=CircuitBreaker(max_consecutive_failures=3),
        )
        return engine, ctx, page, context, sample_result_high

    @pytest.mark.asyncio
    async def test_skips_when_signal_not_matched(self, engine_setup, sample_result_low):
        engine, ctx, page, context, _ = engine_setup
        action = ClickAction("#btn", Payload(jitter_ms=(0, 0), confirm_timeout_ms=100))
        engine.add_action(action)

        results = await engine.evaluate_and_execute(sample_result_low)
        assert results == []

    @pytest.mark.asyncio
    async def test_executes_when_signal_matched(self, engine_setup):
        engine, ctx, page, context, result_high = engine_setup
        element = MagicMock()
        element.is_visible = AsyncMock(return_value=False)
        ctx.protocol.table_frame = MagicMock()
        ctx.protocol.table_frame.query_selector = AsyncMock(return_value=element)
        element.click = AsyncMock()

        action = ClickAction("#btn", Payload(jitter_ms=(0, 0), confirm_timeout_ms=100))
        engine.add_action(action)

        results = await engine.evaluate_and_execute(result_high)
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].confirmation_passed is True
        page.on.assert_called_with("response", engine._on_response)

    @pytest.mark.asyncio
    async def test_halts_when_breaker_trips(self, engine_setup):
        engine, ctx, page, context, result_high = engine_setup
        engine.circuit_breaker.max_consecutive_failures = 1
        action = ClickAction("#btn", Payload(jitter_ms=(0, 0), confirm_timeout_ms=100))
        ctx.protocol.table_frame = MagicMock()
        ctx.protocol.table_frame.query_selector = AsyncMock(return_value=None)
        engine.add_action(action)

        results = await engine.evaluate_and_execute(result_high)
        assert len(results) == 1
        assert results[0].success is False

    @pytest.mark.asyncio
    async def test_on_response_records_403(self, engine_setup):
        engine, ctx, page, context, result_high = engine_setup
        response = MagicMock()
        response.status = 403
        response.url = "http://example.com/api"
        await engine._on_response(response)
        assert len(engine.circuit_breaker.forbidden_timestamps) == 1

    @pytest.mark.asyncio
    async def test_add_action(self, engine_setup):
        engine, _, _, _, _ = engine_setup
        action = ClickAction("#x", Payload(jitter_ms=(0, 0), confirm_timeout_ms=100))
        engine.add_action(action)
        assert len(engine.actions) == 1

# ─────────────────────────────────────────────────────────────────────────────
# Tests: protocol_executors
# ─────────────────────────────────────────────────────────────────────────────

class TestProtocolExecutors:
    def test_ensure_valid_session_loads_cookies(self, tmp_path):
        from protocol_executors import ensure_valid_session

        storage = tmp_path / "storage.json"
        storage.write_text(json.dumps({"cookies": [{"name": "x", "value": "y"}]}))
        context = AsyncMock()
        protocol = MagicMock(_storage_path=str(storage))

        asyncio.get_event_loop().run_until_complete(
            ensure_valid_session(protocol, context, str(storage))
        )
        context.add_cookies.assert_awaited_once_with([{"name": "x", "value": "y"}])

    def test_ensure_valid_session_missing_file(self):
        from protocol_executors import ensure_valid_session

        context = AsyncMock()
        protocol = MagicMock(_storage_path="nonexistent.json")
        asyncio.get_event_loop().run_until_complete(
            ensure_valid_session(protocol, context, "nonexistent.json")
        )
        context.add_cookies.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_action_frame(self):
        from protocol_executors import resolve_action_frame

        protocol = MagicMock(spec=EnvProtocolC)
        frame = MagicMock()
        protocol.find_table_frame = AsyncMock(return_value=(frame, None))
        page = MagicMock()
        result, err = await resolve_action_frame(protocol, page, "hint")
        assert result is frame
        assert err is None

    @pytest.mark.asyncio
    async def test_resolve_action_frame_not_found(self):
        from protocol_executors import resolve_action_frame

        protocol = MagicMock(spec=EnvProtocolC)
        protocol.find_table_frame = AsyncMock(return_value=(None, None))
        page = MagicMock()
        result, err = await resolve_action_frame(protocol, page, "hint")
        assert result is None
        assert "not found" in err