#!/usr/bin/env python3
"""Run Coverage Feature Extraction (CF-SCD steps 1-2) on a real Mentor
POWER/DISTANCE export, instead of synthetic data.

A single export like this only covers one time window (e.g. one hour), so
there isn't yet a multi-day history to build a Baseline / run Spatial
Change Detection / Root Cause Analysis / Coverage Health Score (steps 3-6) -
those need the same export repeated daily over 1-4+ weeks. This script
demonstrates what step 1-2 (Coverage Fingerprint) looks like on genuine
VNPT data: real RSRP, real distances, real UE positions.

With ``--cell-config``, every measured cell in the POWER records is used -
not just the serving cell's EC_0, but every candidate/neighbor cell's
EC_1..EC_11 too - with distance and bearing computed geometrically from
real site coordinates (Lat/Long + azimuth). Without it, only the serving
cell is used, with distance from the DISTANCE record and an estimated
(not exact) site position for bearing.

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
from coverage_intelligence.loader_cell_config import attach_geometry, load_cell_config_xlsx  # noqa: E402
from coverage_intelligence.loader_mentor import (  # noqa: E402
    estimate_site_positions,
    load_mentor_export,
    load_mentor_power_measurements,
)
from coverage_intelligence.real_data_viz import build_fingerprint_report, fig_cell_coverage_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mentor_export", type=str, help="Path to a raw Mentor POWER/DISTANCE export (tab-separated)")
    parser.add_argument("--cell-config", type=str, default=None, help="Optional real cell/site config .xlsx (Latitude/Longitude/azimuth/tilt)")
    parser.add_argument("--out", type=str, default="out_real")
    parser.add_argument("--site-max-distance-m", type=float, default=60.0)
    parser.add_argument("--n-example-cells", type=int, default=6, help="How many per-cell 2-panel heatmaps to render")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.cell_config:
        print(f"[1/4] Parsing {args.mentor_export} - every measured cell "
              "(serving EC_0 AND candidate/neighbor EC_1..EC_11) ...")
        m = load_mentor_power_measurements(args.mentor_export)
        print(f"      -> {len(m):,} power measurements, {m['cell_id'].nunique()} distinct cells "
              f"(serving+candidate), time span {m['timestamp'].min()} .. {m['timestamp'].max()}")

        print(f"[2/4] Parsing {args.cell_config} and computing REAL distance+bearing "
              "for every measurement (site coords + azimuth)...")
        cell_config = load_cell_config_xlsx(args.cell_config)
        ue = attach_geometry(m, cell_config)
        print(f"      -> {len(ue):,}/{len(m):,} measurements matched a cell with known site "
              f"coordinates, covering {ue['cell_id'].nunique()} distinct cells")
    else:
        print(f"[1/4] Parsing {args.mentor_export} (serving cell only; pass --cell-config "
              "to also use candidate/neighbor EC_1..EC_11 readings) ...")
        ue = load_mentor_export(args.mentor_export)
        print(f"      -> {len(ue):,} UE Report samples, {ue['cell_id'].nunique()} distinct serving cells, "
              f"time span {ue['timestamp'].min()} .. {ue['timestamp'].max()}")

        print("[2/4] No --cell-config given: estimating site position from closest-in samples "
              f"(<= {args.site_max_distance_m:.0f} m) instead of using real coordinates...")
        sites = estimate_site_positions(ue, max_distance_m=args.site_max_distance_m)
        ue = ue.merge(sites[["cell_id", "site_x_m", "site_y_m"]], on="cell_id", how="left")
        dx = ue["ue_x_m"] - ue["site_x_m"]
        dy = ue["ue_y_m"] - ue["site_y_m"]
        ue["bearing_deg"] = np.degrees(np.arctan2(dx, dy)) % 360

    n_with_bearing = ue["bearing_deg"].notna().sum()
    n_cells_with_bearing = ue.loc[ue["bearing_deg"].notna(), "cell_id"].nunique()
    print(f"      -> {n_with_bearing:,}/{len(ue):,} samples have a usable bearing, covering "
          f"{n_cells_with_bearing}/{ue['cell_id'].nunique()} cells")

    print("[3/4] Extracting Coverage Fingerprints (Steps 1-2)...")
    # Finer rings than the pipeline's tuned default (config.py) - this only
    # produces a stable, informative heatmap because including candidate/
    # neighbor EC_i readings (see step 1/2 above) pushes per-cell sample
    # density well above what a single day of serving-cell-only UE Report
    # normally has.
    cfg = CFSCDConfig(ring_edges_m=[float(x) for x in range(0, 1001, 50)] + [1500.0, 2000.0, 3000.0, 5000.0])
    fp = compute_fingerprints(ue, cfg)
    print(f"      -> {len(fp.scalar)} (cell, day) fingerprint rows, "
          f"{len(cfg.ring_edges_m)} rings x {cfg.n_direction_sectors} directions "
          f"({len(cfg.ring_edges_m) * cfg.n_direction_sectors} bins)")

    ue.to_csv(os.path.join(args.out, "ue_reports_parsed.csv"), index=False)
    fp.scalar.reset_index().to_csv(os.path.join(args.out, "coverage_fingerprint_scalar.csv"), index=False)
    fp.full_vector().reset_index().to_csv(os.path.join(args.out, "coverage_fingerprint_full.csv"), index=False)

    print("\nTop 15 cells by sample count:")
    cols = ["sample_count", "rsrp_mean", "rsrp_median", "pct_good", "pct_poor", "dist_mean", "dist_p90"]
    print(fp.scalar.sort_values("sample_count", ascending=False)[cols].head(15).to_string())

    if "bearing_deg" in ue.columns and ue["bearing_deg"].notna().any():
        print(f"\n[4/4] Rendering Coverage Fingerprint reports (heatmap + coverage rose + real points, "
              f"plus a full Signal/Distance/Ring/Direction/Grid feature report) for the "
              f"{args.n_example_cells} best-covered cells...")
        rd_dens_cols = [c for c in fp.ring_direction.columns if c.endswith("_density")]
        populated_bins = (fp.ring_direction[rd_dens_cols] > 0).sum(axis=1).sort_values(ascending=False)
        example_cells = [cid for cid, _ in populated_bins.head(args.n_example_cells).index]
        viz_dir = os.path.join(args.out, "cell_fingerprints")
        os.makedirs(viz_dir, exist_ok=True)
        for cell_id in example_cells:
            fig = fig_cell_coverage_report(fp, ue, cell_id, cfg)
            safe_name = cell_id.replace("/", "_")
            out_path = os.path.join(viz_dir, f"{safe_name}.html")
            fig.write_html(out_path)

            report = build_fingerprint_report(fp, cell_id, cfg)
            report_path = os.path.join(viz_dir, f"{safe_name}_features.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"      -> {out_path}  +  {report_path}")
    else:
        print("\n[4/4] No cells have a usable bearing - skipping per-cell heatmaps.")

    print(
        "\nThis is only Coverage Fingerprint extraction (steps 1-2). To run "
        "Baseline / Spatial Change Detection / RCA / Coverage Health Score "
        "(steps 3-6), repeat this same export daily for 1-4+ weeks and feed "
        "the concatenated UE Report tables into "
        "CoverageIntelligencePipeline.run() instead of the synthetic generator."
    )


if __name__ == "__main__":
    main()
