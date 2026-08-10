"""
test_predictability_analyzer.py
-------------------------------
Validation suite for predictability_analyzer.py.

Each test feeds a *known* signal into ``PredictabilityAnalyzer.analyze`` and
checks that the corresponding sub-analysis lights up while the random baseline
stays dark.  Run standalone (no pytest required):

    python test_predictability_analyzer.py

The tests are deterministic (seeded RNG) and assert statistical behaviour,
not exact floats, so they are stable across platforms.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

from predictability_analyzer import (
    PredictabilityAnalyzer,
    _effective_bins,
    _assign_states,
)

PASS = 0
FAIL = 0


def math_close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ── 1. Uniform-random baseline ────────────────────────────────────────────────

def test_uniform_baseline():
    rng = np.random.default_rng(123)
    vals = rng.random(5000)
    r = PredictabilityAnalyzer().analyze(vals)

    check("random score < 0.30", r.predictability_score < 0.30,
          f"score={r.predictability_score:.3f}")
    check("random uniform test passes", r.frequency.is_uniform is True,
          f"p={r.frequency.chi2_pvalue:.3g}")
    check("random no dominant lag", r.autocorrelation.dominant_lag == 0,
          f"lag={r.autocorrelation.dominant_lag}")
    check("random no spurious change points",
          r.change_point.n_significant == 0,
          f"n={r.change_point.n_significant}")
    check("random cond MI ~ 0", r.conditional.normalized_mi < 0.05,
          f"mi={r.conditional.normalized_mi:.3f}")


# ── 2. Periodicity / ACF ────────────────────────────────────────────────────

def test_periodic_signal():
    # A perfectly repeating 50-value pattern tiled 100x -> strong ACF at lag 50.
    pattern = np.linspace(0.0, 1.0, 50, endpoint=False)
    vals = np.tile(pattern, 100)
    r = PredictabilityAnalyzer(value_range=(0.0, 1.0), max_lag=100).analyze(vals)

    check("periodic dominant lag is a multiple of 50",
          r.autocorrelation.dominant_lag > 0
          and r.autocorrelation.dominant_lag % 50 == 0,
          f"lag={r.autocorrelation.dominant_lag}")
    check("periodic strong ACF peak",
          r.autocorrelation.periodicity_score > 0.8,
          f"acf={r.autocorrelation.periodicity_score:.3f}")
    check("periodic ACF at lag 0 == 1", math_close(r.autocorrelation.acf[0], 1.0))
    check("periodic overall score high",
          r.predictability_score > 0.4, f"score={r.predictability_score:.3f}")

# ── 3. Conditional-probability / sequence dependence ─────────────────────────

def test_markov_chain():
    # Strongly persistent Markov chain over 5 states -> high MI at lag 1.
    rng = np.random.default_rng(3)
    n = 6000
    states = np.zeros(n, dtype=int)
    p_stay = 0.85
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < p_stay else rng.integers(0, 5)
    vals = (states + 0.5) / 5.0          # spread into [0.1, 0.9]
    r = PredictabilityAnalyzer().analyze(vals)

    check("markov normalised MI well above random",
          r.conditional.normalized_mi > 0.35,
          f"mi={r.conditional.normalized_mi:.3f}")
    check("transition matrix rows sum to 1",
          np.allclose(r.conditional.transition_matrix.sum(axis=1), 1.0))
    check("transition matrix shape (n_states, n_states)",
          r.conditional.transition_matrix.shape == (10, 10),
          f"shape={r.conditional.transition_matrix.shape}")


# ── 4. Frequency / non-uniformity ────────────────────────────────────────────

def test_non_uniform_gaussian():
    rng = np.random.default_rng(11)
    vals = np.clip(rng.normal(0.5, 0.1, size=5000), 0.0, 1.0)
    r = PredictabilityAnalyzer(value_range=(0.0, 1.0)).analyze(vals)

    check("gaussian flagged non-uniform", r.frequency.is_uniform is False)
    check("gaussian deviation score high",
          r.frequency.deviation_score > 0.3,
          f"dev={r.frequency.deviation_score:.3f}")
    check("gaussian KS statistic high",
          r.frequency.ks_statistic > 0.2,
          f"D={r.frequency.ks_statistic:.3f}")


# ── 5. Change-point detection ───────────────────────────────────────────────

def test_change_point():
    rng = np.random.default_rng(5)
    n = 4000
    half = rng.random(n // 2) * 0.4            # [0, 0.4]
    second = 0.6 + rng.random(n // 2) * 0.4     # [0.6, 1.0]
    vals = np.concatenate([half, second])
    r = PredictabilityAnalyzer(value_range=(0.0, 1.0)).analyze(vals)

    check("change point detected", r.change_point.n_significant >= 1,
          f"n={r.change_point.n_significant}")
    check("change KS statistic very high",
          r.change_point.change_score > 0.5,
          f"D={r.change_point.change_score:.3f}")
    positions = [cp[0] for cp in r.change_point.change_points]
    check("change point near true boundary",
          any(abs(pos - n // 2) < 100 for pos in positions),
          f"positions={positions}")


# ── 6. Ordering: structured > random ───────────────────────────────────────

def test_score_ordering():
    rng = np.random.default_rng(7)
    random_vals = rng.random(4000)
    pattern = np.linspace(0.0, 1.0, 50, endpoint=False)
    periodic_vals = np.tile(pattern, 80)

    r_rand = PredictabilityAnalyzer().analyze(random_vals)
    r_per = PredictabilityAnalyzer(value_range=(0.0, 1.0),
                                   max_lag=100).analyze(periodic_vals)
    check("periodic score > random score",
          r_per.predictability_score > r_rand.predictability_score,
          f"periodic={r_per.predictability_score:.3f} "
          f"random={r_rand.predictability_score:.3f}")

# ── 7. Edge cases ────────────────────────────────────────────────────────────

def test_constant_series():
    r = PredictabilityAnalyzer().analyze([5.0] * 100)
    check("constant does not crash", True)
    check("constant score in [0,1]",
          0.0 <= r.predictability_score <= 1.0)
    check("constant flagged non-uniform", r.frequency.is_uniform is False)
    check("constant no lag", r.autocorrelation.dominant_lag == 0)
    check("constant no change points", r.change_point.n_significant == 0)
    check("constant notes mention zero-variance",
          any("zero variance" in note for note in r.notes),
          f"notes={r.notes}")


def test_too_small():
    r = PredictabilityAnalyzer().analyze([1.0, 2.0, 3.0])
    check("tiny input score 0", r.predictability_score == 0.0)
    check("tiny input notes present", len(r.notes) > 0)


def test_empty_input():
    r = PredictabilityAnalyzer().analyze([])
    check("empty input score 0", r.predictability_score == 0.0)
    check("empty input no crash", True)


# ── 8. Helpers ───────────────────────────────────────────────────────────────

def test_helpers():
    check("effective bins not above requested", _effective_bins(100, 10) == 10)
    check("effective bins capped at 50", _effective_bins(1000, 200) == 50)
    s = _assign_states(np.array([0.0, 0.35, 0.99]), 0.0, 1.0, 10)
    check("assign_states shape", s.shape == (3,), f"shape={s.shape}")
    check("assign_states in range",
          bool(np.all((s >= 0) & (s < 10))), f"states={s}")


# ── 9. CLI end-to-end ────────────────────────────────────────────────────────

def test_cli():
    from predictability_analyzer import main
    with tempfile.TemporaryDirectory() as td:
        import pandas as pd
        rng = np.random.default_rng(42)
        rows = []
        for i in range(2000):
            rows.append((f"2026-08-09T01:33:{i % 60:02d}",
                         "CPU_SPIKE", float(rng.random())))
            rows.append((f"2026-08-09T01:34:{i % 60:02d}",
                         "MEMORY_PRESSURE", float(rng.random())))
        df = pd.DataFrame(rows, columns=["Timestamp", "Event Name",
                                         "Numeric Result"])
        df.to_csv(os.path.join(td, "dashboard_data_cycle0001.csv"), index=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--data-dir", td, "--event", "CPU_SPIKE",
                       "--max-lag", "30"])
        out = buf.getvalue()
        check("cli exit code 0", rc == 0, f"rc={rc}")
        check("cli prints score", "Predictability Score" in out)


# ── runner ─────────────────────────────────────────────────────────────────

TESTS = [
    test_uniform_baseline,
    test_periodic_signal,
    test_markov_chain,
    test_non_uniform_gaussian,
    test_change_point,
    test_score_ordering,
    test_constant_series,
    test_too_small,
    test_empty_input,
    test_helpers,
    test_cli,
]


def run_all():
    print("=" * 68)
    print("predictability_analyzer test suite")
    print("=" * 68)
    for fn in TESTS:
        print(f"\n[{fn.__name__}]")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            global FAIL
            FAIL += 1
            print(f"  [ERROR] {fn.__name__} raised "
                  f"{type(exc).__name__}: {exc}")
    print("\n" + "=" * 68)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 68)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run_all())
