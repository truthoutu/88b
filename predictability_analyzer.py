"""
predictability_analyzer.py
--------------------------
Statistical analysis suite for detecting micro-patterns and periodicity in
time-series data harvested from distributed sources (e.g. the output of
``scraper.py``).

The module treats an incoming batch of numerical values as a candidate stream
from a Pseudo-Random Number Generator (PRNG) and answers the question:

    "How non-random / predictable has this window of data become?"

It emits a single **Predictability Score** in [0.0, 1.0] together with a rich,
diagnostic breakdown of four complementary analyses:

  1. Frequency Distribution Analysis  – deviation from a uniform law
  2. Autocorrelation Testing (ACF)    – repeating lag / PRNG cycle
  3. Conditional Probability Mapping  – P(X occurs after Y) transition matrix
  4. Change Point Detection           – regime / algorithm / security shifts

Dependencies
────────────
Only NumPy + SciPy are required.  ``statsmodels`` is *not* a project
dependency, so the ACF and the change-point scan are implemented from scratch
on top of ``scipy.stats`` primitives.

Public API
──────────
    >>> from predictability_analyzer import PredictabilityAnalyzer
    >>> ana = PredictabilityAnalyzer()
    >>> result = ana.analyze(values)          # values: 1-D sequence of floats
    >>> result.predictability_score           # float in [0, 1]
    >>> result.autocorrelation.dominant_lag

CLI usage
─────────
    python predictability_analyzer.py                            # every event
    python predictability_analyzer.py --event CPU_SPIKE          # one event
    python predictability_analyzer.py --data-dir ./scraped_data --bins 12 --max-lag 200
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Defaults & scoring rubric
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BINS = 10            # histogram bins for frequency / conditional analysis
DEFAULT_MAX_LAG_CAP = 500    # auto max_lag is min(n // 2, this cap)
DEFAULT_ALPHA = 0.05         # significance level for *all* hypothesis tests
DEFAULT_VALUE_RANGE: Optional[tuple[float, float]] = None   # auto -> data min/max
DEFAULT_WINDOW_CAP = 100     # sliding window size for change-point scan
DEFAULT_WEIGHTS: dict[str, float] = {
    "frequency": 0.25,
    "autocorrelation": 0.30,
    "conditional": 0.30,
    "change_point": 0.15,
}

# Human-readable meaning of the final score.
SCORE_INTERPRETATION: list[tuple[float, float, str]] = [
    (0.00, 0.30, "essentially random"),
    (0.30, 0.60, "mildly structured"),
    (0.60, 1.01, "strongly predictable / non-random"),
]

# Minimum sample size before a sub-analysis is declared "insufficient data".
_MIN_CHI2 = 20      # need enough mass for a reliable chi-square
_MIN_ACF = 10       # need a handful of lags to speak of periodicity
_MIN_COND = 20      # need transitions to populate a matrix
_MIN_CHANGE = 40    # need two windows to compare


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrequencyAnalysis:
    """Outcome of the uniformity (frequency-distribution) test."""

    observed_counts: np.ndarray
    expected_per_bin: float
    chi2_stat: float
    chi2_pvalue: float
    cohen_w: float            # raw effect size = sqrt(chi2 / n)
    deviation_score: float     # Cohen's w normalised to [0, 1]
    ks_statistic: float       # Kolmogorov-Smirnov D vs uniform
    ks_pvalue: float
    is_uniform: bool          # True iff the chi-square test fails to reject H0


@dataclass
class AutocorrelationResult:
    """Outcome of the ACF / periodicity scan."""

    acf: np.ndarray                       # length max_lag + 1 (index 0 == 1.0)
    confidence_band: float                # +/- Bartlett 1-alpha band magnitude
    significant_lags: list[int]           # lags where |acf| > band
    dominant_lag: int                     # lag of strongest positive peak (0 if none)
    dominant_acf: float                   # ACF value at dominant_lag
    periodicity_score: float               # = dominant_acf, in [0, 1] (0 if no cycle)


@dataclass
class ConditionalProbabilityResult:
    """Outcome of the transition-matrix / sequence scan."""

    n_states: int
    transition_matrix: np.ndarray          # (n_states, n_states) P(next=j | cur=i)
    mutual_information: float             # MI between consecutive states (bits)
    normalized_mi: float                   # MI / log2(n_states) in [0, 1]
    transition_score: float                # mean row TV-distance-from-uniform [0, 1]
    strongest_state: int                   # bin index with the most deterministic row


@dataclass
class ChangePointResult:
    """Outcome of the regime-change scan."""

    change_points: list[tuple[int, float, float]]   # (position, ks_D, pvalue)
    max_ks_statistic: float               # strongest D observed across the scan
    n_significant: int                    # number of non-overlapping shifts kept
    change_score: float                    # max D among significant changes (0 if none)


@dataclass
class PredictabilityResult:
    """Aggregate result returned by ``PredictabilityAnalyzer.analyze``."""

    n_samples: int
    weights: dict[str, float]

    frequency: FrequencyAnalysis
    autocorrelation: AutocorrelationResult
    conditional: ConditionalProbabilityResult
    change_point: ChangePointResult

    sub_scores: dict[str, float]            # each independently in [0, 1]
    predictability_score: float           # weighted mean, in [0, 1]
    interpretation: str
    notes: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _effective_bins(n: int, requested: int) -> int:
    """Pick the number of histogram bins, never exceeding n // 5 (chi-square
    rule of thumb: >= 5 expected count per bin)."""
    k = min(requested, max(2, n // 5))
    return min(k, 50)


def _assign_states(values: np.ndarray, vmin: float, vmax: float,
                   k: int) -> np.ndarray:
    """Map each value to a state in [0, k-1] via equal-width bins over
    ``[vmin, vmax]``.  Values outside the range are clamped to the edge bins."""
    edges = np.linspace(vmin, vmax, k + 1)
    states = np.digitize(values, edges[1:-1], right=False).astype(np.intp)
    np.clip(states, 0, k - 1, out=states)
    return states


def _acf_band(alpha: float, n: int) -> float:
    """Two-sided Bartlett confidence-band magnitude for the ACF under the null
    hypothesis of white noise:  +/- z_{1-alpha/2} / sqrt(n)."""
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return z / math.sqrt(n)


def _interpret(score: float) -> str:
    for lo, hi, label in SCORE_INTERPRETATION:
        if lo <= score < hi:
            return label
    return SCORE_INTERPRETATION[-1][2]

# ─────────────────────────────────────────────────────────────────────────────
# Core analyser
# ─────────────────────────────────────────────────────────────────────────────

class PredictabilityAnalyzer:
    """Stateless-in-spirit statistical suite.

    Parameters
    ────────
    n_bins:
        Number of equal-width histogram bins used by the frequency and
        conditional-probability analyses.
    max_lag:
        Largest lag at which to evaluate the ACF.  ``None`` => auto
        (``min(n // 2, 500)``).
    alpha:
        Significance level for every hypothesis test (ACF band, chi-square,
        KS two-sample, change-point scan).
    value_range:
        The *intended* range of the data (e.g. ``(0.0, 1.0)`` for raw PRNG
        output).  When supplied, uniform / binning tests use this range
        rather than the data's own min/max, which matters: a stream that
        only ever produces values in [0.4, 0.6] is *highly* non-uniform if
        it claims to span [0, 1].
    window:
        Half-window size (in samples) for the change-point sliding scan.
        ``None`` => auto (``min(n // 4, 100)``).
    weights:
        Per-component weights for the aggregate Predictability Score.  They
        are normalised internally so they always sum to 1.
    """

    def __init__(
        self,
        n_bins: int = DEFAULT_BINS,
        max_lag: Optional[int] = None,
        alpha: float = DEFAULT_ALPHA,
        value_range: Optional[tuple[float, float]] = DEFAULT_VALUE_RANGE,
        window: Optional[int] = None,
        weights: Optional[dict[str, float]] = None,
        bonferroni: bool = True,
    ) -> None:
        self.n_bins = max(2, n_bins)
        self.max_lag = max_lag
        self.alpha = alpha
        self.value_range = value_range
        self.window = window
        self.bonferroni = bool(bonferroni)
        w = dict(weights if weights is not None else DEFAULT_WEIGHTS)
        total = sum(w.values()) or 1.0
        self.weights: dict[str, float] = {k: v / total for k, v in w.items()}

    def _eff_alpha(self, n_tests: int) -> float:
        """Per-test alpha with a Bonferroni correction for the multi-lag
        ACF scan and the multi-position change-point scan.

        Single-shot tests (chi-square goodness-of-fit, KS uniformity) use
        ``self.alpha`` untouched, so a small sample is never denied a
        legitimate uniformity rejection just because few tests ran.
        """
        if self.bonferroni and n_tests > 1:
            return self.alpha / n_tests
        return self.alpha

    # ── public ------------------------------------------------─────────────

    def analyze(self, values: Sequence[float]) -> PredictabilityResult:
        """Run every sub-analysis on a 1-D sequence of floats and return a
        fully populated :class:`PredictabilityResult`."""
        arr = np.asarray(values, dtype=float).ravel()
        n = arr.size
        notes: list[str] = []

        if n == 0:
            notes.append("Empty input: nothing to analyse.")
            empty_freq = FrequencyAnalysis(
                np.array([]), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, True)
            empty_acf = AutocorrelationResult(
                np.array([1.0]), 0.0, [], 0, 0.0, 0.0)
            empty_cond = ConditionalProbabilityResult(
                0, np.array([]), 0.0, 0.0, 0.0, -1)
            empty_chg = ChangePointResult([], 0.0, 0, 0.0)
            return PredictabilityResult(
                n_samples=0, weights=dict(self.weights),
                frequency=empty_freq, autocorrelation=empty_acf,
                conditional=empty_cond, change_point=empty_chg,
                sub_scores={k: 0.0 for k in self.weights},
                predictability_score=0.0, interpretation=_interpret(0.0),
                notes=notes,
            )

        # Resolve the value range used for binning / uniform testing.
        if self.value_range is not None:
            vmin = float(self.value_range[0])
            vmax = float(self.value_range[1])
        else:
            vmin = float(arr.min())
            vmax = float(arr.max())
        if vmax == vmin:
            vmax = vmin + 1.0
            notes.append("All observed values identical; treating span as "
                         "1.0 for binning.")

        k = _effective_bins(n, self.n_bins)

        freq = self._frequency_analysis(arr, n, k, vmin, vmax, notes)
        acf = self._autocorrelation_analysis(arr, n, notes)
        cond = self._conditional_analysis(arr, n, k, vmin, vmax, notes)
        chg = self._change_point_analysis(arr, n, notes)

        sub_scores: dict[str, float] = {
            "frequency": freq.deviation_score,
            "autocorrelation": acf.periodicity_score,
            "conditional": cond.normalized_mi,
            "change_point": chg.change_score,
        }
        pred = sum(self.weights[name] * sub_scores[name]
                   for name in sub_scores)
        pred = float(np.clip(pred, 0.0, 1.0))

        # Cross-analysis caveat: heavy positive autocorrelation (a strong
        # repeating cycle) inflates the count of locally-significant
        # change points, because adjacent sliding windows can diverge
        # merely due to clustered runs rather than a genuine regime shift.
        if acf.periodicity_score > 0.5 and chg.n_significant > 5:
            notes.append(
                "High autocorrelation coincides with many change points; "
                "some may reflect local clustering rather than true "
                "regime shifts. Cross-check periodicity via the dominant "
                "ACF lag."
            )

        return PredictabilityResult(
            n_samples=n,
            weights=dict(self.weights),
            frequency=freq,
            autocorrelation=acf,
            conditional=cond,
            change_point=chg,
            sub_scores=sub_scores,
            predictability_score=pred,
            interpretation=_interpret(pred),
            notes=notes,
        )

    # ── 1. Frequency distribution analysis ────────────────────────────────

    def _frequency_analysis(
        self, arr: np.ndarray, n: int, k: int,
        vmin: float, vmax: float, notes: list[str],
    ) -> FrequencyAnalysis:
        """Chi-square goodness-of-fit for uniformity + KS cross-check.

        Cohen's *w* (= sqrt(chi2 / n)) measures effect size independently of
        sample size; it is normalised by its theoretical maximum sqrt(k-1) so
        the resulting ``deviation_score`` lives in [0, 1].
        """
        if n < _MIN_CHI2 or k < 2:
            notes.append("Frequency analysis skipped: insufficient data "
                         f"(n={n}, k={k}).")
            empty = np.zeros(1)
            return FrequencyAnalysis(
                observed_counts=empty, expected_per_bin=0.0,
                chi2_stat=0.0, chi2_pvalue=1.0, cohen_w=0.0,
                deviation_score=0.0, ks_statistic=0.0, ks_pvalue=1.0,
                is_uniform=True,
            )

        # Histogram over [vmin, vmax] (uniform null expectation).
        counts, _ = np.histogram(arr, bins=k, range=(vmin, vmax))
        expected = n / k
        # chi2 goodness-of-fit against *equal* frequencies (uniform).
        chi2_stat, chi2_p = stats.chisquare(counts, f_exp=expected)
        cohen_w = math.sqrt(chi2_stat / n) if n else 0.0
        deviation_score = float(np.clip(cohen_w / math.sqrt(k - 1), 0.0, 1.0))

        # KS cross-check: rescale to [0,1] and test against U(0,1).
        span = vmax - vmin
        rescaled = (arr - vmin) / span
        ks_res = stats.kstest(rescaled, "uniform", args=(0.0, 1.0))
        ks_stat = float(ks_res.statistic)
        ks_p = float(ks_res.pvalue)

        is_uniform = bool(chi2_p > self.alpha)

        return FrequencyAnalysis(
            observed_counts=counts,
            expected_per_bin=float(expected),
            chi2_stat=float(chi2_stat),
            chi2_pvalue=chi2_p,
            cohen_w=float(cohen_w),
            deviation_score=deviation_score,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p,
            is_uniform=is_uniform,
        )

    # ── 2. Autocorrelation / periodicity ────────────────────────────────────

    def _autocorrelation_analysis(self, arr: np.ndarray, n: int,
                                  notes: list[str]) -> AutocorrelationResult:
        """Normalised sample ACF + Bartlett significance band.

        r_k = SUM (x_t - xbar)(x_{t+k} - xbar) / SUM (x_t - xbar)^2

        Under H0 (iid white noise) each r_k ~ N(0, 1/n), giving the familiar
        +/- 1.96/sqrt(n) band.  A strong *positive* peak at a specific lag is
        the signature of a PRNG cycle / repeating pattern.
        """
        max_lag = self.max_lag or min(n // 2, DEFAULT_MAX_LAG_CAP)
        max_lag = max(1, min(max_lag, n - 1))

        if n < _MIN_ACF or max_lag < 1:
            notes.append("Autocorrelation analysis skipped: insufficient "
                         f"data (n={n}, max_lag={max_lag}).")
            acf = np.zeros(max(1, max_lag + 1))
            return AutocorrelationResult(
                acf=acf, confidence_band=0.0, significant_lags=[],
                dominant_lag=0, dominant_acf=0.0, periodicity_score=0.0,
            )

        mean = arr.mean()
        demeaned = arr - mean
        c0 = float(np.dot(demeaned, demeaned))   # SUM (x_t - xbar)^2 == n*var
        if c0 <= 0.0:
            notes.append("Autocorrelation skipped: zero variance in window.")
            acf = np.zeros(max_lag + 1)
            acf[0] = 1.0
            return AutocorrelationResult(
                acf=acf, confidence_band=0.0, significant_lags=[],
                dominant_lag=0, dominant_acf=0.0, periodicity_score=0.0,
            )

        acf = np.empty(max_lag + 1)
        acf[0] = 1.0
        for lag in range(1, max_lag + 1):
            ck = float(np.dot(demeaned[: n - lag], demeaned[lag:]))
            acf[lag] = ck / c0

        band = _acf_band(self._eff_alpha(max_lag), n)
        significant_lags = [int(l) for l in range(1, max_lag + 1)
                            if abs(acf[l]) > band]
        positive_sig = [l for l in significant_lags if acf[l] > band]

        dominant_lag = 0
        dominant_acf = 0.0
        periodicity_score = 0.0
        if positive_sig:
            dominant_lag = int(max(positive_sig, key=lambda l: acf[l]))
            dominant_acf = float(acf[dominant_lag])
            periodicity_score = float(np.clip(dominant_acf, 0.0, 1.0))
        else:
            notes.append(
                "No statistically significant positive ACF peak detected "
                "(no obvious PRNG cycle in this window)."
            )
            neg_sig = [l for l in significant_lags if acf[l] < -band]
            if neg_sig:
                notes.append(
                    f"Significant negative autocorrelation at lag(s) "
                    f"{neg_sig[:8]} - suggests alternation, not a repeat."
                )

        return AutocorrelationResult(
            acf=acf,
            confidence_band=band,
            significant_lags=significant_lags,
            dominant_lag=dominant_lag,
            dominant_acf=dominant_acf,
            periodicity_score=periodicity_score,
        )

    # ── 3. Conditional probability mapping ────────────────────────────────

    def _conditional_analysis(
        self, arr: np.ndarray, n: int, k: int,
        vmin: float, vmax: float, notes: list[str],
    ) -> ConditionalProbabilityResult:
        """Build the P(next state | current state) transition matrix and
        quantify sequential structure with normalised mutual information.

        MI = SUM P(i,j) * log2[ P(i,j) / (P(i) * P(j)) ]

        Under independence MI = 0; the strongest possible coupling (perfectly
        deterministic transitions) gives MI = log2(k), hence
        ``normalized_mi`` in [0, 1].  ``transition_score`` is the mean
        total-variation distance of each row from the uniform row, a more
        sample-stable complement to MI.
        """
        if n < _MIN_COND or k < 2:
            notes.append("Conditional analysis skipped: insufficient data "
                         f"(n={n}, k={k}).")
            return ConditionalProbabilityResult(
                n_states=k,
                transition_matrix=np.full((k, k), 1.0 / k),
                mutual_information=0.0,
                normalized_mi=0.0,
                transition_score=0.0,
                strongest_state=-1,
            )

        states = _assign_states(arr, vmin, vmax, k)
        cur = states[:-1]
        nxt = states[1:]

        # Raw joint counts, then row-normalise -> P(next | cur).
        joint = np.zeros((k, k), dtype=float)
        np.add.at(joint, (cur, nxt), 1.0)
        row_sums = joint.sum(axis=1, keepdims=True)
        # Row-normalise safely: unseen rows get a uniform row (no divide-by-zero warning).
        rs = row_sums.ravel()
        transition = np.full((k, k), 1.0 / k)
        transition[rs > 0] = joint[rs > 0] / rs[rs > 0][:, None]

        # Marginals.
        p_cur = row_sums.ravel()
        total = p_cur.sum()
        p_cur = p_cur / total if total else p_cur
        p_next = (joint.sum(axis=0)) / total if total else np.full(k, 1.0 / k)

        # Mutual information (bits), guarding zero joint entries.
        joint_prob = transition * p_cur[:, None]      # P(i, j) = P(j|i) * P(i)
        denom = p_cur[:, None] * p_next[None, :]      # P(i) * P(j)  (k, k)
        mask = joint_prob > 0.0
        mi = float(np.sum(
            joint_prob[mask] * np.log2(joint_prob[mask] / denom[mask])
        ))
        max_mi = math.log2(k)
        normalized_mi = float(np.clip(mi / max_mi, 0.0, 1.0)) if max_mi > 0 else 0.0

        # Per-row total-variation distance from uniform, then average.
        uniform_row = np.full(k, 1.0 / k)
        tv_per_row = 0.5 * np.abs(transition - uniform_row).sum(axis=1)
        transition_score = float(np.clip(tv_per_row.mean(), 0.0, 1.0))
        strongest_state = int(np.argmax(tv_per_row))

        return ConditionalProbabilityResult(
            n_states=k,
            transition_matrix=transition,
            mutual_information=mi,
            normalized_mi=normalized_mi,
            transition_score=transition_score,
            strongest_state=strongest_state,
        )

    # ── 4. Change-point detection ──────────────────────────────────────────

    def _change_point_analysis(self, arr: np.ndarray, n: int,
                               notes: list[str]) -> ChangePointResult:
        """Sliding-window two-sample KS scan for distributional regime shifts.

        For every candidate split point *c* the windows [c-w, c) and
        [c, c+w) are compared with ``scipy.stats.ks_2samp``.  Candidate
        change points are then greedily clustered (kept only if they are
        >= *w* apart) to avoid reporting clumps of correlated detections.
        """
        window = self.window or min(n // 4, DEFAULT_WINDOW_CAP)
        window = max(2, min(window, n // 2 - 1))

        if n < _MIN_CHANGE or window < 2:
            notes.append("Change-point analysis skipped: insufficient data "
                         f"(n={n}, window={window}).")
            return ChangePointResult(
                change_points=[], max_ks_statistic=0.0,
                n_significant=0, change_score=0.0,
            )

        candidates: list[tuple[int, float, float]] = []
        for c in range(window, n - window):
            left = arr[c - window: c]
            right = arr[c: c + window]
            ks_res = stats.ks_2samp(left, right)
            candidates.append((int(c), float(ks_res.statistic),
                               float(ks_res.pvalue)))

        max_d = max((d for _, d, _ in candidates), default=0.0)

        # Keep only statistically significant shifts, then cluster.
        significant = [c for c in candidates if c[2] < self._eff_alpha(len(candidates))]
        significant.sort(key=lambda t: t[1], reverse=True)  # strongest first
        kept: list[tuple[int, float, float]] = []
        for pos, d, p in significant:
            if any(abs(pos - kp[0]) < window for kp in kept):
                continue          # too close to an already-kept peak
            kept.append((pos, d, p))
        kept.sort()

        change_score = float(max((d for _, d, _ in kept), default=0.0))
        if not kept:
            notes.append("No statistically significant regime change detected.")

        return ChangePointResult(
            change_points=kept,
            max_ks_statistic=max_d,
            n_significant=len(kept),
            change_score=change_score,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CSV convenience (CLI glue)
# ─────────────────────────────────────────────────────────────────────────────

def _load_event_values(data_dir: Path, event: Optional[str],
                       event_col: str, value_col: str) -> np.ndarray:
    """Load every ``dashboard_data_*.csv`` snapshot, optionally filter to a
    single event, and return the numeric column as a float array.

    Header names are normalised to lower-case so ``Event Name`` /
    ``Numeric Result`` map cleanly even if a dashboard spells them
    ``event_name`` / ``value``.
    """
    import pandas as pd  # pandas is a declared project dependency

    files = sorted(data_dir.glob("dashboard_data_*.csv"))
    if not files:
        files = sorted(data_dir.glob("*.csv"))              # fallback
    if not files:
        return np.array([], dtype=float)

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    if not frames:
        return np.array([], dtype=float)

    table = pd.concat(frames, ignore_index=True)
    norm = {c.lower(): c for c in table.columns}
    event_col = norm.get((event_col or "event name").lower(), event_col)
    value_col = norm.get((value_col or "numeric result").lower(), value_col)

    if event is not None:
        table = table[table[event_col].astype(str) == event]
    values = pd.to_numeric(table[value_col], errors="coerce").dropna()
    return np.asarray(values, dtype=float)


def _print_report(result: PredictabilityResult) -> None:
    """Pretty-print a full diagnostic report with rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console(force_terminal=True)

    sc = result.sub_scores
    panel = Panel.fit(
        f"[bold]Predictability Score[/bold]: [cyan]{result.predictability_score:.3f}[/cyan]\n"
        f"[dim]{result.interpretation}[/dim]  "
        f"({result.n_samples} samples)",
        border_style="bright_blue",
    )
    console.print(panel)

    # Sub-score breakdown.
    t = Table(title="Per-analysis contribution", show_header=True,
              header_style="bold magenta")
    t.add_column("Component", style="bold")
    t.add_column("Weight", justify="right")
    t.add_column("Sub-score", justify="right")
    t.add_column("Interpretation")
    labels = {
        "frequency": "uniform-law deviation",
        "autocorrelation": "periodicity / cycle strength",
        "conditional": "sequence (P(X|Y)) dependence",
        "change_point": "regime-shift magnitude",
    }
    for name in result.weights:
        s = sc[name]
        if s >= 0.6:
            lbl = f"[red]{labels[name]} - strong[/red]"
        elif s >= 0.3:
            lbl = f"[yellow]{labels[name]} - moderate[/yellow]"
        else:
            lbl = f"[green]{labels[name]} - low[/green]"
        t.add_row(name, f"{result.weights[name]:.2f}", f"{s:.3f}", lbl)
    console.print(t)

    # Frequency distribution table.
    fq = result.frequency
    if len(fq.observed_counts) > 0 and fq.expected_per_bin > 0:
        ft = Table(title="Frequency distribution (chi-square uniformity test)",
                   show_header=True, header_style="bold magenta")
        ft.add_column("Bin #", justify="right")
        ft.add_column("Observed", justify="right")
        ft.add_column("Expected", justify="right")
        for i, obs in enumerate(fq.observed_counts):
            ft.add_row(str(i + 1), str(int(obs)),
                       f"{fq.expected_per_bin:.1f}")
        console.print(ft)
        console.print(
            f"  chi2 = {fq.chi2_stat:.2f}  p = {fq.chi2_pvalue:.3g}  "
            f"Cohen's w = {fq.cohen_w:.3f}  "
            f"KS D = {fq.ks_statistic:.3f} (p={fq.ks_pvalue:.3g})  "
            f"uniform? {fq.is_uniform}"
        )

    # Autocorrelation summary.
    ac = result.autocorrelation
    console.print("\n[bold]Autocorrelation (ACF)[/bold]")
    if ac.dominant_lag:
        console.print(
            f"  Dominant cycle lag = [cyan]{ac.dominant_lag}[/cyan]  "
            f"(ACF = {ac.dominant_acf:.3f})  band +/-{ac.confidence_band:.3f}"
        )
    else:
        console.print("  No significant repeating lag detected.")
    n_sig = len(ac.significant_lags)
    console.print(f"  Significant lags: {n_sig}  "
                  f"-> {[l for l in ac.significant_lags[:12]]}"
                  f"{' ...' if n_sig > 12 else ''}")

    # Conditional transition matrix.
    cd = result.conditional
    console.print("\n[bold]Conditional probability (P(next | cur))[/bold]")
    console.print(f"  States: {cd.n_states}  MI = {cd.mutual_information:.3f} bits  "
                  f"normalised MI = {cd.normalized_mi:.3f}  "
                  f"transition TV = {cd.transition_score:.3f}")
    if cd.n_states and cd.n_states <= 16:
        mt = Table(title="Transition matrix  P(j | i)")
        mt.add_column("from\\to", justify="right")
        for j in range(cd.n_states):
            mt.add_column(str(j), justify="right")
        for i in range(cd.n_states):
            mt.add_row(str(i), *[f"{v:.2f}" for v in cd.transition_matrix[i]])
        console.print(mt)
    elif cd.n_states:
        console.print(f"  (matrix {cd.n_states}x{cd.n_states} too large)")

    # Change-point summary.
    cp = result.change_point
    console.print("\n[bold]Change-point detection[/bold]")
    console.print(f"  Shifts detected: {cp.n_significant}  "
                  f"max KS D = {cp.max_ks_statistic:.3f}  "
                  f"score = {cp.change_score:.3f}")
    for pos, d, p in cp.change_points:
        console.print(f"    t ~= {pos}  D={d:.3f}  p={p:.3g}")

    if result.notes:
        console.print("\n[dim]Notes:[/dim]")
        for note in result.notes:
            console.print(f"  [dim]- {note}[/dim]")

# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Statistical predictability suite for harvested time-series data. "
            "Computes frequency/uniformity, autocorrelation cycle, conditional "
            "transition matrix, and change-point analyses, then emits a single "
            "Predictability Score in [0, 1]."
        )
    )
    p.add_argument("--data-dir", default="./scraped_data")
    p.add_argument("--event", default=None,
                   help="Filter to a single Event Name (e.g. CPU_SPIKE).")
    p.add_argument("--event-col", default="Event Name")
    p.add_argument("--value-col", default="Numeric Result")
    p.add_argument("--bins", type=int, default=DEFAULT_BINS,
                   help="Histogram bins for frequency/conditional analysis.")
    p.add_argument("--max-lag", type=int, default=None,
                   help="Maximum ACF lag (default: min(n//2, 500)).")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="Significance level for all hypothesis tests.")
    p.add_argument("--value-range", type=float, nargs=2, default=None,
                   metavar=("MIN", "MAX"),
                   help="Intended value range, e.g. --value-range 0 1.")
    p.add_argument("--window", type=int, default=None,
                   help="Sliding-window half-size for change-point scan.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_cli().parse_args(argv)
    data_dir = Path(args.data_dir)

    values = _load_event_values(data_dir, args.event, args.event_col,
                                args.value_col)
    if values.size == 0:
        from loguru import logger
        logger.error("No numeric values found in {0}", data_dir)
        print(f"[predictability] No data found in {data_dir.resolve()}")
        return 1

    value_range = (float(args.value_range[0]), float(args.value_range[1])) \
        if args.value_range else None
    analyzer = PredictabilityAnalyzer(
        n_bins=args.bins,
        max_lag=args.max_lag,
        alpha=args.alpha,
        value_range=value_range,
        window=args.window,
    )
    result = analyzer.analyze(values)
    _print_report(result)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
