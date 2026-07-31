#!/usr/bin/env python3
"""Run Coverage Feature Extraction (CF-SCD steps 1-2) on a real Mentor
POWER/DISTANCE export, instead of synthetic data.

A single export like this only covers one time window (e.g. one hour), so
there isn't yet a multi-day history to build a Baseline / run Spatial
Change Detection / Root Cause Analysis / Coverage Health Score (steps 3-6) -
those need the same export repeated daily over 1-4+ weeks. This script
demonstrates what step 1-2 (Coverage Fingerprint) looks like on genuine
VNPT data: real RSRP, real distances, real UE positions - and, when a real
cell config export is supplied, real bearing (from real site coordinates +
azimuth) instead of an estimated one.

Usage:
    python scripts/run_real_data_demo.py path/to/raw_mentor.txt --out out_real/
    python scripts/run_real_data_demo.py path/to/raw_mentor.txt \\
        --cell-config path/to/cell_config.xlsx --out out_real/
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from coverage_intelligence.config import CFSCDConfig  # noqa: E402
from coverage_intelligence.features import compute_fingerprints  # noqa: E402
from coverage_intelligence.loader_cell_config import attach_bearing, load_cell_config_xlsx  # noqa: E402
from coverage_intelligence.loader_mentor import estimate_site_positions, load_mentor_export  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mentor_export", type=str, help="Path to a raw Mentor POWER/DISTANCE export (tab-separated)")
    parser.add_argument("--cell-config", type=str, default=None, help="Optional real cell/site config .xlsx (Latitude/Longitude/azimuth/tilt)")
    parser.add_argument("--out", type=str, default="out_real")
    parser.add_argument("--site-max-distance-m", type=float, default=60.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/3] Parsing {args.mentor_export} ...")
    ue = load_mentor_export(args.mentor_export)
    print(f"      -> {len(ue):,} UE Report samples, {ue['cell_id'].nunique()} distinct serving cells, "
          f"{ue['site_id'].nunique()} distinct sites, time span "
          f"{ue['timestamp'].min()} .. {ue['timestamp'].max()}")

    if args.cell_config:
        print(f"[2/3] Parsing {args.cell_config} and attaching REAL bearing (site coords + azimuth)...")
        cell_config = load_cell_config_xlsx(args.cell_config)
        n_matched = ue["cell_id"].isin(cell_config["cell_id"]).sum()
        print(f"      -> {cell_config['cell_id'].nunique()} cells in config, "
              f"{ue['cell_id'].isin(cell_config['cell_id']).groupby(ue['cell_id']).any().sum()} of the export's "
              f"{ue['cell_id'].nunique()} cells matched by exact cell_id")
        ue = attach_bearing(ue, cell_config, fallback_max_distance_m=args.site_max_distance_m)
    else:
        print("[2/3] No --cell-config given: estimating site position from closest-in samples "
              f"(<= {args.site_max_distance_m:.0f} m) instead of using real coordinates...")
        sites = estimate_site_positions(ue, max_distance_m=args.site_max_distance_m)
        ue = ue.merge(sites[["cell_id", "site_x_m", "site_y_m"]], on="cell_id", how="left")
        dx = ue["ue_x_m"] - ue["site_x_m"]
        dy = ue["ue_y_m"] - ue["site_y_m"]
        ue["bearing_deg"] = np.degrees(np.arctan2(dx, dy)) % 360

    n_with_bearing = ue["bearing_deg"].notna().sum()
    n_cells_with_bearing = ue.loc[ue["bearing_deg"].notna(), "cell_id"].nunique()
    print(f"      -> {n_with_bearing:,}/{len(ue):,} samples have a usable bearing, covering "
          f"{n_cells_with_bearing}/{ue['cell_id'].nunique()} cells (the rest still count toward "
          "Signal/Distance/Ring features, which don't need bearing)")

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
