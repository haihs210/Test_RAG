#!/usr/bin/env python3
"""Run the full CF-SCD pipeline on synthetic data and write an alerts CSV +
an HTML dashboard.

Example:
    python scripts/run_demo.py --n-cells 200 --n-days 35 \\
        --samples-per-cell-day 1500 --out out/

For a "large sample data" style run (multi-million UE Report rows):
    python scripts/run_demo.py --n-cells 500 --n-days 30 \\
        --samples-per-cell-day 3000 --out out/
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coverage_intelligence import CFSCDConfig, CoverageIntelligencePipeline  # noqa: E402
from coverage_intelligence.dashboard import build_dashboard  # noqa: E402
from coverage_intelligence.synthetic import generate_demo_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cells", type=int, default=200)
    parser.add_argument("--n-days", type=int, default=35)
    parser.add_argument("--samples-per-cell-day", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/3] Generating synthetic UE Report / RF config / Alarm data "
          f"({args.n_cells} cells x {args.n_days} days x ~{args.samples_per_cell_day} samples/cell/day)...")
    t0 = time.time()
    cells, scenarios, ue_reports, alarms = generate_demo_dataset(
        n_cells=args.n_cells,
        n_days=args.n_days,
        base_samples_per_cell_day=args.samples_per_cell_day,
        seed=args.seed,
    )
    print(f"      -> {len(ue_reports):,} UE Report rows, {len(cells)} cells, {len(alarms)} alarms "
          f"({time.time() - t0:.1f}s)")

    cells.to_csv(os.path.join(args.out, "cells.csv"), index=False)
    ue_reports.head(5000).to_csv(os.path.join(args.out, "ue_reports_preview.csv"), index=False)

    print("[2/3] Running the CF-SCD pipeline (feature extraction -> baseline -> "
          "detection -> RCA -> Coverage Health Score)...")
    t0 = time.time()
    pipeline = CoverageIntelligencePipeline(CFSCDConfig())
    result = pipeline.run(ue_reports, cells, alarms)
    print(f"      -> {len(result.alerts)} alerts on {result.fingerprints.scalar.index.get_level_values('cell_id').nunique()} cells "
          f"({time.time() - t0:.1f}s)")

    alerts_path = os.path.join(args.out, "alerts.csv")
    result.alerts.to_csv(alerts_path, index=False)
    print(f"      -> alerts written to {alerts_path}")

    print("[3/3] Building HTML dashboard...")
    dash_path = os.path.join(args.out, "dashboard.html")
    build_dashboard(result, pipeline.cfg, dash_path)
    print(f"      -> dashboard written to {dash_path}")

    if not result.alerts.empty:
        print("\nTop 10 most critical alerts:")
        cols = [c for c in ["date", "cell_id", "province", "band", "chs", "chs_class", "top_cause", "recommendation"] if c in result.alerts.columns]
        print(result.alerts.sort_values("chs")[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
