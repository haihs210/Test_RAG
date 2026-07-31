"""Step 6 of CF-SCD: Coverage Health Score (CHS).

A single 0-100 index combining signal quality, fingerprint stability,
spatial-change magnitude, root-cause severity and customer impact, then
classified into Excellent / Good / Warning / Critical and aggregated (traffic
weighted) up the cluster / province / network hierarchy - matching:

    "CHS con duoc tong hop theo nhieu cap quan ly (cum cell, huyen, tinh,
    toan mang) co trong so theo luu luong va muc do anh huong."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFSCDConfig

CAUSE_SEVERITY = {
    "OUTAGE": 1.0,
    "TILT_POWER_DRIFT": 0.75,
    "AZIMUTH_DRIFT": 0.6,
    "SHADOWING": 0.5,
    "OVERSHOOT": 0.35,
}


def classify(chs: pd.Series, cfg: CFSCDConfig) -> pd.Series:
    bins = [-np.inf, cfg.chs_warning, cfg.chs_good, cfg.chs_excellent, np.inf]
    labels = ["Critical", "Warning", "Good", "Excellent"]
    return pd.cut(chs, bins=bins, labels=labels, right=False)


def compute_health_score(
    scalar: pd.DataFrame,
    detection: pd.DataFrame,
    rca_df: pd.DataFrame,
    cfg: CFSCDConfig,
) -> pd.DataFrame:
    idx = scalar.index
    out = pd.DataFrame(index=idx)

    # 1) Signal quality: where does rsrp_mean sit between the poor/good thresholds
    span = cfg.rsrp_good_dbm - cfg.rsrp_poor_dbm
    signal_quality = ((scalar["rsrp_mean"] - cfg.rsrp_poor_dbm) / span).clip(0, 1)
    out["signal_quality"] = signal_quality * 100

    # 2) Fingerprint stability: from the Spatial Similarity Index
    ssi = detection["ssi"].reindex(idx)
    out["fingerprint_stability"] = ssi.clip(lower=0, upper=1).fillna(1.0) * 100

    # 3) Spatial change magnitude: composite of the three rule-based deltas
    rsrp_drop_norm = (detection["rsrp_drop_db"].reindex(idx) / cfg.rule_rsrp_drop_db).clip(0, 1.5)
    radius_norm = (detection["radius_change_pct"].reindex(idx).abs() / cfg.rule_radius_change_pct).clip(0, 1.5)
    sample_norm = (detection["sample_drop_pct"].reindex(idx) / cfg.rule_sample_drop_pct).clip(0, 1.5)
    severity = (rsrp_drop_norm.fillna(0) + radius_norm.fillna(0) + sample_norm.fillna(0)) / 3
    severity = severity.clip(0, 1)
    out["spatial_change_magnitude"] = (1 - severity) * 100

    # 4) RCA severity: worse when a high-confidence, high-severity cause is diagnosed
    rca_severity = pd.Series(0.0, index=idx)
    if not rca_df.empty:
        rca_indexed = rca_df.set_index(["cell_id", "date"])
        cause_weight = rca_indexed["top_cause"].map(CAUSE_SEVERITY).fillna(0.4)
        rca_severity_vals = (cause_weight * rca_indexed["top_confidence"]).clip(0, 1)
        rca_severity.loc[rca_severity_vals.index.intersection(idx)] = rca_severity_vals
    out["rca_severity"] = (1 - rca_severity) * 100

    # 5) Impact: traffic-weighted - an anomaly on a high-traffic cell hurts more
    typical = scalar.groupby(level="date")["sample_count"].transform("median").replace(0, np.nan)
    traffic_factor = (scalar["sample_count"] / typical).clip(0.2, 2.0).fillna(1.0) / 2.0
    out["impact"] = (1 - severity.clip(0, 1) * traffic_factor) * 100

    w = cfg.chs_weights
    out["chs"] = (
        w["signal_quality"] * out["signal_quality"]
        + w["fingerprint_stability"] * out["fingerprint_stability"]
        + w["spatial_change_magnitude"] * out["spatial_change_magnitude"]
        + w["rca_severity"] * out["rca_severity"]
        + w["impact"] * out["impact"]
    ).clip(0, 100)
    out["chs_class"] = classify(out["chs"], cfg)
    return out


def aggregate_health_score(
    chs_df: pd.DataFrame,
    scalar: pd.DataFrame,
    cells: pd.DataFrame,
    level_col: str,
) -> pd.DataFrame:
    """Traffic-weighted CHS rollup to cluster / province / network level."""
    merged = chs_df.join(scalar[["sample_count"]])
    merged = merged.reset_index().merge(cells[["cell_id", level_col]], on="cell_id", how="left")
    merged["weighted_chs"] = merged["chs"] * merged["sample_count"]

    grouped = merged.groupby([level_col, "date"]).agg(
        chs=("weighted_chs", "sum"),
        total_samples=("sample_count", "sum"),
        n_cells=("cell_id", "nunique"),
        n_critical=("chs_class", lambda s: int((s == "Critical").sum())),
        n_warning=("chs_class", lambda s: int((s == "Warning").sum())),
    )
    grouped["chs"] = grouped["chs"] / grouped["total_samples"].replace(0, np.nan)
    grouped["chs_class"] = classify(grouped["chs"], CFSCDConfig())
    return grouped.drop(columns="total_samples")
