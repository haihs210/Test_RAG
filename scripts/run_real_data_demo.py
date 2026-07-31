#!/usr/bin/env python3
"""Run Coverage Feature Extraction (CF-SCD steps 1-2) on a real Mentor
POWER/DISTANCE export, instead of synthetic data.

A single export like this only covers one time window (here: one hour), so
there isn't yet a multi-day history to build a Baseline / run Spatial
Change Detection / Root Cause Analysis / Coverage Health Score (steps 3-6) -
those need the same export repeated daily over 1-4+ weeks. This script
demonstrates what step 1-2 (Coverage Fingerprint) looks like on genuine
VNPT data: real RSRP, real distances, real UE positions.

Usage:
    python scripts/run_real_data_demo.py path/to/raw_mentor.txt --out out_real/
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coverage_intelligence.config import CFSCDConfig  # noqa: E402
from coverage_intelligence.features import compute_fingerprints  # noqa: E402
from coverage_intelligence.loader_mentor import estimate_site_positions, load_mentor_export  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mentor_export", type=str, help="Path to a raw Mentor POWER/DISTANCE export (tab-separated)")
    parser.add_argument("--out", type=str, default="out_real")
    parser.add_argument("--site-max-distance-m", type=float, default=60.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/3] Parsing {args.mentor_export} ...")
    ue = load_mentor_export(args.mentor_export)
    print(f"      -> {len(ue):,} UE Report samples, {ue['cell_id'].nunique()} distinct serving cells, "
          f"{ue['site_id'].nunique()} distinct sites, time span "
          f"{ue['timestamp'].min()} .. {ue['timestamp'].max()}")

    print("[2/3] Estimating site positions from closest-in samples "
          f"(<= {args.site_max_distance_m:.0f} m) and deriving bearing...")
    sites = estimate_site_positions(ue, max_distance_m=args.site_max_distance_m)
    ue = ue.merge(sites[["cell_id", "site_x_m", "site_y_m"]], on="cell_id", how="left")
    dx = ue["ue_x_m"] - ue["site_x_m"]
    dy = ue["ue_y_m"] - ue["site_y_m"]
    ue["bearing_deg"] = np.degrees(np.arctan2(dx, dy)) % 360
    n_with_bearing = ue["bearing_deg"].notna().sum()
    print(f"      -> {len(sites)} cells got an estimated site position "
          f"({n_with_bearing:,}/{len(ue):,} samples have a usable bearing; "
          "the rest still count toward Signal/Distance/Ring features)")

    print("[3/3] Extracting Coverage Fingerprints (Steps 1-2)...")
    cfg = CFSCDConfig()
    fp = compute_fingerprints(ue, cfg)
    print(f"      -> {len(fp.scalar)} (cell, day) fingerprint rows")

    ue.to_csv(os.path.join(args.out, "ue_reports_parsed.csv"), index=False)
    fp.scalar.reset_index().to_csv(os.path.join(args.out, "coverage_fingerprint_scalar.csv"), index=False)
    fp.full_vector().reset_index().to_csv(os.path.join(args.out, "coverage_fingerprint_full.csv"), index=False)

    print("\nTop 15 cells by sample count:")
    cols = ["sample_count", "rsrp_mean", "rsrp_median", "pct_good", "pct_poor", "dist_mean", "dist_p90"]
    print(fp.scalar.sort_values("sample_count", ascending=False)[cols].head(15).to_string())

    print(
        "\nThis is only Coverage Fingerprint extraction (steps 1-2). To run "
        "Baseline / Spatial Change Detection / RCA / Coverage Health Score "
        "(steps 3-6), repeat this same export daily for 1-4+ weeks and feed "
        "the concatenated UE Report tables into "
        "CoverageIntelligencePipeline.run() instead of the synthetic generator."
    )


if __name__ == "__main__":
    main()
