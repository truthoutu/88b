"""
analyze.py
----------
Post-scrape analysis utility.  Loads all CSVs produced by scraper.py from the
output directory, merges them, and prints summary statistics plus a simple
ASCII time-series chart useful for quick longitudinal inspection.

Usage:
    python analyze.py
    python analyze.py --data-dir ./scraped_data --event CPU_SPIKE
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict
import statistics


DEFAULT_DATA_DIR = Path("./scraped_data")


def load_all(data_dir: Path) -> list[dict]:
    """Load every CSV in *data_dir* and return merged rows as dicts."""
    files = sorted(data_dir.glob("dashboard_data_*.csv"))
    if not files:
        print(f"[analyze] No CSV files found in {data_dir.resolve()}")
        return []

    all_rows: list[dict] = []
    for f in files:
        with f.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                all_rows.append(row)

    print(f"[analyze] Loaded {len(all_rows):,} rows from {len(files)} file(s).")
    return all_rows


def summarise(rows: list[dict], event_filter: str | None = None) -> None:
    """Print per-event summary statistics."""
    if event_filter:
        rows = [r for r in rows if r["Event Name"] == event_filter]
        if not rows:
            print(f"[analyze] No rows for event '{event_filter}'.")
            return

    by_event: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            val = float(row["Numeric Result"])
            by_event[row["Event Name"]].append(val)
        except (ValueError, KeyError):
            continue

    print(f"\n{'Event Name':<30} {'Count':>6} {'Min':>10} {'Max':>10} {'Mean':>10} {'StdDev':>10}")
    print("-" * 80)
    for ev in sorted(by_event):
        vals = by_event[ev]
        count = len(vals)
        lo    = min(vals)
        hi    = max(vals)
        mean  = statistics.mean(vals)
        std   = statistics.stdev(vals) if count > 1 else 0.0
        print(f"{ev:<30} {count:>6} {lo:>10.3f} {hi:>10.3f} {mean:>10.3f} {std:>10.3f}")


def ascii_chart(rows: list[dict], event: str, width: int = 60) -> None:
    """Draw a primitive ASCII sparkline for a single event."""
    vals = []
    for r in rows:
        if r["Event Name"] == event:
            try:
                vals.append(float(r["Numeric Result"]))
            except ValueError:
                pass

    if not vals:
        print(f"[analyze] No data for event '{event}'.")
        return

    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    BARS = "_.-=+*#@"  # ASCII-safe 8-level sparkline (works on all terminals)
    normalized = [int((v - lo) / span * (len(BARS) - 1)) for v in vals[-width:]]
    chart = "".join(BARS[n] for n in normalized)

    print(f"\n  {event}  (last {len(normalized)} samples, range [{lo:.2f} – {hi:.2f}])")
    print(f"  {chart}")


def main():
    p = argparse.ArgumentParser(description="Analyze scraped dashboard CSV files.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--event", default=None, help="Filter to a specific Event Name.")
    p.add_argument("--chart", default=None, help="Draw ASCII sparkline for this event.")
    args = p.parse_args()

    rows = load_all(Path(args.data_dir))
    if not rows:
        return

    summarise(rows, event_filter=args.event)

    chart_event = args.chart or args.event
    if chart_event:
        ascii_chart(rows, chart_event)


if __name__ == "__main__":
    import sys, os
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
