"""Step 4 of CF-SCD: Spatial Change Detection.

Three complementary detectors, combined by Ensemble Voting so an alert only
fires when several methods agree - this is what keeps the false-alarm rate
down:

  * Rule-based Detection      - large, obvious level/reach/volume shifts
  * Spatial Similarity Index  - shape change of the Ring x Direction map
                                 even when the aggregate mean is unchanged
  * Time-series Detection     - EWMA + CUSUM + a lightweight STL-style
                                 trend/residual check, for slow drifts
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baseline import BaselineResult
from .config import CFSCDConfig


@dataclass
class EffectiveBaseline:
    median: pd.DataFrame
    mad: pd.DataFrame
    is_valid: pd.Series
    source: pd.Series  # "own" | "peer" | "none"


def merge_baselines(own: BaselineResult, peer: BaselineResult) -> EffectiveBaseline:
    """Fall back to peer-group (same band/tech) baseline when a cell's own
    trailing history is too short - e.g. a newly commissioned cell in the
    Overshoot scenario, which needs an accurate evaluation even without
    history of its own yet.
    """
    use_own = own.is_valid.reindex(own.median.index, fill_value=False)
    median = own.median.where(use_own, peer.median)
    mad = own.mad.where(use_own, peer.mad)
    is_valid = use_own | peer.is_valid.reindex(median.index, fill_value=False)
    source = pd.Series(
        np.where(use_own, "own", np.where(peer.is_valid.reindex(median.index, fill_value=False), "peer", "none")),
        index=median.index,
    )
    return EffectiveBaseline(median=median, mad=mad, is_valid=is_valid, source=source)


def rule_based_flags(scalar: pd.DataFrame, baseline: EffectiveBaseline, cfg: CFSCDConfig) -> pd.DataFrame:
    idx = scalar.index
    med = baseline.median.reindex(idx)
    valid = baseline.is_valid.reindex(idx, fill_value=False)

    rsrp_drop = med["rsrp_mean"] - scalar["rsrp_mean"]
    radius_change_signed = (scalar["dist_p90"] - med["dist_p90"]) / med["dist_p90"].replace(0, np.nan)
    radius_change = radius_change_signed.abs()
    sample_drop = (med["sample_count"] - scalar["sample_count"]) / med["sample_count"].replace(0, np.nan)

    out = pd.DataFrame(index=idx)
    out["rsrp_drop_db"] = rsrp_drop
    out["radius_change_pct"] = radius_change
    out["radius_change_signed"] = radius_change_signed
    out["sample_drop_pct"] = sample_drop
    out["rule_flag"] = valid & (
        (rsrp_drop >= cfg.rule_rsrp_drop_db)
        | (radius_change.fillna(0) >= cfg.rule_radius_change_pct)
        | (sample_drop.fillna(0) >= cfg.rule_sample_drop_pct)
    )
    return out


def compute_ssi(ring_direction: pd.DataFrame, baseline: EffectiveBaseline, cfg: CFSCDConfig) -> pd.DataFrame:
    """Density-weighted correlation between the current and baseline
    Ring x Direction RSRP maps - i.e. does the *shape* of the fingerprint
    still match, independent of any overall level shift.
    """
    mean_cols = [c for c in ring_direction.columns if c.endswith("_mean")]
    dens_cols = [c for c in ring_direction.columns if c.endswith("_density")]

    idx = ring_direction.index
    curr_mean = ring_direction[mean_cols].reindex(idx).to_numpy(dtype=float)
    base_mean = baseline.median.reindex(idx)[mean_cols].to_numpy(dtype=float)
    base_dens = baseline.median.reindex(idx)[dens_cols].to_numpy(dtype=float)

    weight = np.nan_to_num(base_dens, nan=0.0)
    valid_bin = ~np.isnan(curr_mean) & ~np.isnan(base_mean)
    weight = np.where(valid_bin, weight, 0.0)
    wsum = weight.sum(axis=1, keepdims=True)
    wsum_safe = np.where(wsum == 0, 1.0, wsum)
    w = weight / wsum_safe

    cm = np.nan_to_num(curr_mean)
    bm = np.nan_to_num(base_mean)
    wmean_c = (w * cm).sum(axis=1, keepdims=True)
    wmean_b = (w * bm).sum(axis=1, keepdims=True)
    dc = cm - wmean_c
    db = bm - wmean_b
    cov = (w * dc * db).sum(axis=1)
    var_c = (w * dc**2).sum(axis=1)
    var_b = (w * db**2).sum(axis=1)
    denom = np.sqrt(var_c * var_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        ssi = np.where(denom > 1e-9, cov / denom, np.nan)
    ssi = np.clip(ssi, -1.0, 1.0)

    out = pd.DataFrame(index=idx)
    out["ssi"] = ssi
    valid = baseline.is_valid.reindex(idx, fill_value=False) & (wsum.ravel() > 0)
    out["ssi_flag"] = valid & (out["ssi"].fillna(1.0) < cfg.ssi_alert_threshold)
    return out


def time_series_flags(scalar: pd.DataFrame, baseline: EffectiveBaseline, cfg: CFSCDConfig) -> pd.DataFrame:
    """EWMA + CUSUM + a lightweight trend/residual ("STL-lite") check on the
    daily RSRP mean series of each cell, catching slow multi-day drifts that
    a single-day rule threshold would miss.
    """
    metric = "rsrp_mean"
    out = pd.DataFrame(index=scalar.index, columns=["ewma_flag", "cusum_flag", "stl_flag"], dtype=bool)
    out[:] = False

    for cell_id, sub in scalar[[metric]].groupby(level=0):
        sub = sub.droplevel(0).sort_index()
        dates = sub.index.to_numpy()
        x = sub[metric].to_numpy(dtype=float)
        med = baseline.median.reindex(list(zip([cell_id] * len(dates), dates)))[metric].to_numpy(dtype=float)
        mad = baseline.mad.reindex(list(zip([cell_id] * len(dates), dates)))[metric].to_numpy(dtype=float)
        valid = baseline.is_valid.reindex(list(zip([cell_id] * len(dates), dates)), fill_value=False).to_numpy()
        mad_safe = np.where((mad > 1e-6) & ~np.isnan(mad), mad, np.nan)

        ewma = np.full(len(x), np.nan)
        cusum_pos = np.full(len(x), 0.0)
        cusum_neg = np.full(len(x), 0.0)
        ewma_flag = np.zeros(len(x), dtype=bool)
        cusum_flag = np.zeros(len(x), dtype=bool)
        stl_flag = np.zeros(len(x), dtype=bool)

        prev_ewma = None
        for i in range(len(x)):
            prev_ewma = x[i] if prev_ewma is None else cfg.ewma_alpha * x[i] + (1 - cfg.ewma_alpha) * prev_ewma
            ewma[i] = prev_ewma
            if valid[i] and not np.isnan(mad_safe[i]):
                z_ewma = (prev_ewma - med[i]) / mad_safe[i]
                ewma_flag[i] = abs(z_ewma) > cfg.ewma_z_threshold

                dev = (x[i] - med[i]) / mad_safe[i]
                cusum_pos[i] = max(0.0, (cusum_pos[i - 1] if i > 0 else 0.0) + dev - cfg.cusum_k)
                cusum_neg[i] = min(0.0, (cusum_neg[i - 1] if i > 0 else 0.0) + dev + cfg.cusum_k)
                cusum_flag[i] = (cusum_pos[i] > cfg.cusum_h) or (cusum_neg[i] < -cfg.cusum_h)

                start = max(0, i - cfg.stl_window)
                if i - start >= 3:
                    trend = np.nanmean(x[start:i])
                    resid_z = (x[i] - trend) / mad_safe[i]
                    stl_flag[i] = abs(resid_z) > cfg.stl_z_threshold
            else:
                cusum_pos[i] = cusum_pos[i - 1] if i > 0 else 0.0
                cusum_neg[i] = cusum_neg[i - 1] if i > 0 else 0.0

        keys = list(zip([cell_id] * len(dates), dates))
        out.loc[keys, "ewma_flag"] = ewma_flag
        out.loc[keys, "cusum_flag"] = cusum_flag
        out.loc[keys, "stl_flag"] = stl_flag

    out["ts_votes"] = out[["ewma_flag", "cusum_flag", "stl_flag"]].sum(axis=1)
    out["ts_flag"] = out["ts_votes"] >= 1
    return out


def ensemble_vote(rule_df: pd.DataFrame, ssi_df: pd.DataFrame, ts_df: pd.DataFrame, cfg: CFSCDConfig) -> pd.DataFrame:
    out = rule_df.join(ssi_df).join(ts_df)
    out["votes"] = (
        out["rule_flag"].astype(int) + out["ssi_flag"].astype(int) + out["ts_flag"].astype(int)
    )
    out["is_anomaly"] = out["votes"] >= cfg.ensemble_votes_required
    return out
