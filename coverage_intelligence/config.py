"""Tunable parameters for the CF-SCD pipeline.

Every threshold below is a starting point taken from the ranges the proposal
mentions (e.g. a 7-30 day baseline window). They are meant to be recalibrated
per market with the Human-in-the-loop feedback described in step 7.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CFSCDConfig:
    # --- Coverage Fingerprint geometry (Step 1-2) ---
    # This default is tuned for the daily-aggregate sample density the
    # synthetic pipeline (and a production baseline/detection run) works
    # with. Finer rings need proportionally more samples per bin to stay
    # stable (see docs/algorithm_analysis.md) - scripts/run_real_data_demo.py
    # uses a finer ~50m ring config instead, since including candidate/
    # neighbor EC_i readings (not just the serving cell's EC_0) gives it
    # much higher per-cell sample density than a single day of production
    # UE Report normally would.
    ring_edges_m: List[float] = field(
        default_factory=lambda: [0, 200, 500, 1000, 2000, 5000]
    )
    n_direction_sectors: int = 12  # 30 degrees per sector
    grid_cell_m: float = 500.0
    grid_half_extent_m: float = 2500.0  # grid covers [-extent, +extent] on x/y

    # RSRP quality buckets used by the Signal Feature (dBm thresholds)
    rsrp_good_dbm: float = -95.0
    rsrp_poor_dbm: float = -110.0

    # --- Baseline construction (Step 3) ---
    baseline_window_days: int = 14  # within the 7-30 day range from the proposal
    min_baseline_days: int = 5
    mad_scale: float = 1.4826  # scales MAD to be comparable to a std-dev

    # --- Spatial Change Detection (Step 4) ---
    rule_rsrp_drop_db: float = 6.0          # rule-based: mean RSRP drop
    rule_radius_change_pct: float = 0.30    # rule-based: effective radius change
    rule_sample_drop_pct: float = 0.60      # rule-based: outage-style sample collapse
    ssi_alert_threshold: float = 0.985      # below this similarity -> spatial change
    ewma_alpha: float = 0.3
    ewma_z_threshold: float = 3.0
    cusum_k: float = 0.5   # slack, in baseline MAD units
    cusum_h: float = 5.0   # decision interval, in baseline MAD units
    stl_window: int = 7
    stl_z_threshold: float = 3.0
    ensemble_votes_required: int = 2  # out of {rule, ssi, time-series}

    # --- Root Cause Analysis (Step 5) ---
    n_neighbors: int = 6
    neighbor_radius_m: float = 1500.0
    neighbor_corr_threshold: float = 0.6  # shared-anomaly ratio -> "wide-area"

    # --- Coverage Health Score (Step 6) ---
    chs_weights: dict = field(
        default_factory=lambda: {
            "signal_quality": 0.30,
            "fingerprint_stability": 0.25,
            "spatial_change_magnitude": 0.25,
            "rca_severity": 0.10,
            "impact": 0.10,
        }
    )
    chs_excellent: float = 85.0
    chs_good: float = 70.0
    chs_warning: float = 50.0  # below this -> Critical
