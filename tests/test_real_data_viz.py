import numpy as np
import pandas as pd

from coverage_intelligence.config import CFSCDConfig
from coverage_intelligence.features import compute_fingerprints
from coverage_intelligence.real_data_viz import build_fingerprint_report, fig_cell_coverage_report


def _sample_geometry_df():
    rng = np.random.default_rng(0)
    n = 200
    distance = rng.uniform(10, 400, size=n)
    bearing = rng.uniform(0, 360, size=n)
    rsrp = -90 + rng.normal(0, 5, size=n)
    return pd.DataFrame(
        {
            "cell_id": ["CELL_A"] * n,
            "date": [pd.Timestamp("2026-02-21")] * n,
            "distance_m": distance,
            "bearing_deg": bearing,
            "rsrp_dbm": rsrp,
            "ue_x_m": 1000.0 + distance * np.sin(np.radians(bearing)),
            "ue_y_m": 2000.0 + distance * np.cos(np.radians(bearing)),
            "site_x_m": 1000.0,
            "site_y_m": 2000.0,
            "azimuth_deg": 45.0,
        }
    )


def test_bin_confidence_grows_with_sample_count():
    cfg = CFSCDConfig(confidence_half_count=5.0)
    ue = _sample_geometry_df()
    fp = compute_fingerprints(ue, cfg)
    ring_row = fp.ring.loc["CELL_A"].iloc[0]

    counted_bins = [i for i in range(len(cfg.ring_edges_m)) if ring_row.get(f"ring_{i}_count", 0) > 0]
    assert counted_bins
    for i in counted_bins:
        n = ring_row[f"ring_{i}_count"]
        conf = ring_row[f"ring_{i}_confidence"]
        assert 0 < conf <= 1
        assert abs(conf - n / (n + cfg.confidence_half_count)) < 1e-9

    # a bin with more samples should never have lower confidence
    best = max(counted_bins, key=lambda i: ring_row[f"ring_{i}_count"])
    worst = min(counted_bins, key=lambda i: ring_row[f"ring_{i}_count"])
    if ring_row[f"ring_{best}_count"] > ring_row[f"ring_{worst}_count"]:
        assert ring_row[f"ring_{best}_confidence"] >= ring_row[f"ring_{worst}_confidence"]


def test_fingerprint_report_contains_all_five_feature_groups():
    cfg = CFSCDConfig()
    ue = _sample_geometry_df()
    fp = compute_fingerprints(ue, cfg)
    report = build_fingerprint_report(fp, "CELL_A", cfg)

    for section in ["Signal Feature", "Distance Feature", "Ring Feature", "Direction Feature", "Grid Feature"]:
        assert section in report
    assert "do_tin_cay" in report  # confidence is surfaced, not just the mean


def test_fig_cell_coverage_report_has_three_panels():
    cfg = CFSCDConfig()
    ue = _sample_geometry_df()
    fp = compute_fingerprints(ue, cfg)
    fig = fig_cell_coverage_report(fp, ue, "CELL_A", cfg)

    trace_types = {t.type for t in fig.data}
    assert "heatmap" in trace_types
    assert "barpolar" in trace_types
    assert "scatter" in trace_types
