"""Step 5 of CF-SCD: Root Cause Analysis & Recommendation.

For every anomalous (cell, date) pair, this module:
  1. Looks at how neighboring cells behaved the same day (localized vs
     wide-area, e.g. a power-grid or network-change event).
  2. Scores the fingerprint change pattern against a small expert rule
     library covering the algorithm's named failure-mode scenarios
     (Outage, Shadowing, Tilt/Power drift, Azimuth drift, Overshoot).
  3. Boosts the matching cause when a corroborating Alarm exists.

The rule library is intentionally plain Python (see ``_score_patterns``,
``RECOMMENDATIONS``, ``ALARM_BOOST``) so it is the artifact that gets
edited as Step 7's Human-in-the-loop feedback accumulates - tuning the
expert rule set as verified outcomes come in, without touching the
detection code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import CFSCDConfig
from .detection import EffectiveBaseline

RECOMMENDATIONS = {
    "OUTAGE": "Kiem tra nguon dien / RRU / truyen dan tai tram; dieu phoi xu ly khan cap.",
    "SHADOWING": "Khao sat thuc dia huong bi che chan; theo doi tien trien, can nhac dieu chinh tilt/cong suat hoac bo sung tram.",
    "TILT_POWER_DRIFT": "Kiem tra cau hinh tilt/cong suat phat so voi thiet ke; doi chieu lich thay doi tham so gan nhat.",
    "AZIMUTH_DRIFT": "Kiem tra co hoc anten (huong lap dat thuc te so voi cau hinh); danh gia anh huong thoi tiet/va cham.",
    "OVERSHOOT": "Doi chieu voi nhom cell cung dai/tan/cong nghe; can nhac dieu chinh tilt/cong suat de giam nhieu chong lan.",
}

ALARM_BOOST = {
    "CELL_OUTAGE": ("OUTAGE", 0.35),
    "TX_POWER_LOW": ("TILT_POWER_DRIFT", 0.35),
}


@dataclass
class NeighborIndex:
    cell_to_neighbors: Dict[str, List[str]]


def build_neighbor_index(cells: pd.DataFrame, cfg: CFSCDConfig) -> NeighborIndex:
    sites = cells.drop_duplicates("site_id")[["site_id", "x_m", "y_m"]].reset_index(drop=True)
    tree = cKDTree(sites[["x_m", "y_m"]].to_numpy())
    site_neighbors: Dict[str, List[str]] = {}
    for i, row in sites.iterrows():
        dist, idx = tree.query([row["x_m"], row["y_m"]], k=min(cfg.n_neighbors + 1, len(sites)))
        idx = np.atleast_1d(idx)
        dist = np.atleast_1d(dist)
        keep = [sites.loc[j, "site_id"] for j, d in zip(idx, dist) if d <= cfg.neighbor_radius_m and sites.loc[j, "site_id"] != row["site_id"]]
        site_neighbors[row["site_id"]] = keep

    cells_by_site = cells.groupby("site_id")["cell_id"].apply(list).to_dict()
    cell_to_neighbors: Dict[str, List[str]] = {}
    for _, row in cells.iterrows():
        neighbor_sites = site_neighbors.get(row["site_id"], [])
        neigh_cells: List[str] = []
        for s in neighbor_sites:
            neigh_cells.extend(cells_by_site.get(s, []))
        cell_to_neighbors[row["cell_id"]] = neigh_cells
    return NeighborIndex(cell_to_neighbors=cell_to_neighbors)


def _wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def _score_patterns(
    row: pd.Series,
    peak_shift_deg: float,
    dir_imbalance_delta: float,
    frac_degraded: float,
    frac_improved: float,
    baseline_source: str,
) -> Dict[str, float]:
    rsrp_drop = max(row.get("rsrp_drop_db", 0.0) or 0.0, 0.0)
    radius_change_signed = row.get("radius_change_signed", 0.0) or 0.0
    radius_change = abs(radius_change_signed)
    sample_drop = max(row.get("sample_drop_pct", 0.0) or 0.0, 0.0)
    ssi = row.get("ssi", 1.0)
    ssi = 1.0 if pd.isna(ssi) else ssi
    shape_changed = 1.0 - max(min(ssi, 1.0), -1.0)  # 0 = identical shape, up to 2

    scores: Dict[str, float] = {}

    scores["OUTAGE"] = float(np.clip(0.7 * min(sample_drop / 0.6, 1.5) + 0.3 * min(rsrp_drop / 15.0, 1.5), 0, 1))

    # Shadowing: one or more sectors go bad, the rest of the fingerprint is
    # untouched - critically, no sector *improves* (pure blockage, nothing
    # is redirected there), which is what separates it from Azimuth drift.
    localized_shape_change = 0.05 < frac_degraded < 0.6 and radius_change < 0.20 and sample_drop < 0.3
    no_gain_elsewhere = frac_improved < 0.05
    scores["SHADOWING"] = float(
        np.clip(
            0.45 * min(frac_degraded / 0.3, 1.2)
            + 0.25 * min(rsrp_drop / 10.0, 1.2)
            + 0.2 * (1 if localized_shape_change else 0)
            + 0.1 * (1 if no_gain_elsewhere else 0),
            0,
            1,
        )
    )

    # Tilt/power drift shows up as radius *shrinking* together with an RSRP drop.
    radius_change_ref = cfg_default.rule_radius_change_pct
    shrink = max(-radius_change_signed, 0.0)
    scores["TILT_POWER_DRIFT"] = float(
        np.clip(0.5 * min(shrink / radius_change_ref, 1.5) + 0.5 * min(rsrp_drop / 8.0, 1.5), 0, 1)
    )

    # Azimuth drift: signal is *redistributed*, not just lost - some sectors
    # improve as the beam points elsewhere, others degrade as it leaves them.
    scores["AZIMUTH_DRIFT"] = float(
        np.clip(
            0.4 * min(frac_improved / 0.2, 1.2)
            + 0.3 * min(abs(peak_shift_deg) / 45.0, 1.2)
            + 0.2 * min(max(dir_imbalance_delta, 0) / 10.0, 1.2)
            + 0.1 * min(shape_changed / 0.4, 1.2),
            0,
            1,
        )
    )

    # Overshoot shows up as radius *growing* well beyond the (own or peer-group)
    # baseline - most often on cells too new to have their own history yet.
    growth = max(radius_change_signed, 0.0)
    peer_bonus = 0.15 if baseline_source == "peer" else 0.0
    scores["OVERSHOOT"] = float(np.clip(0.75 * min(growth / radius_change_ref, 1.5) + peer_bonus, 0, 1))

    return scores


# populated by diagnose_alerts() before scoring; kept module-level so
# _score_patterns can reference the run's threshold without a bigger refactor
cfg_default = CFSCDConfig()


def diagnose_alerts(
    alert_keys: List[tuple],
    detection: pd.DataFrame,
    scalar: pd.DataFrame,
    direction: pd.DataFrame,
    direction_baseline: EffectiveBaseline,
    cells: pd.DataFrame,
    alarms: pd.DataFrame,
    neighbors: NeighborIndex,
    anomaly_lookup: Dict[tuple, bool],
    cfg: CFSCDConfig,
) -> pd.DataFrame:
    global cfg_default
    cfg_default = cfg

    n_dirs = cfg.n_direction_sectors
    sector_width = 360.0 / n_dirs
    mean_cols = [f"dir_{i}_mean" for i in range(n_dirs)]
    dens_cols = [f"dir_{i}_density" for i in range(n_dirs)]

    rows = []
    for cell_id, date in alert_keys:
        key = (cell_id, date)
        if key not in detection.index:
            continue
        det_row = detection.loc[key]
        scal_row = scalar.loc[key] if key in scalar.index else pd.Series(dtype=float)

        curr_dir = direction.loc[key, mean_cols] if key in direction.index else pd.Series(np.nan, index=mean_cols)
        curr_dens = direction.loc[key, dens_cols] if key in direction.index else pd.Series(0.0, index=dens_cols)
        base_dir = direction_baseline.median.loc[key, mean_cols] if key in direction_baseline.median.index else pd.Series(np.nan, index=mean_cols)

        curr_valid = curr_dens.to_numpy() > 0
        peak_shift_deg = 0.0
        dir_imbalance_delta = 0.0
        frac_degraded = 0.0
        frac_improved = 0.0
        if curr_valid.any() and not curr_dir.isna().all() and not base_dir.isna().all():
            curr_arr = curr_dir.to_numpy(dtype=float)
            base_arr = base_dir.to_numpy(dtype=float)
            valid_both = curr_valid & ~np.isnan(curr_arr) & ~np.isnan(base_arr)
            if valid_both.any():
                curr_peak = np.nanargmax(np.where(valid_both, curr_arr, -np.inf))
                base_peak = np.nanargmax(np.where(valid_both, base_arr, -np.inf))
                peak_shift_deg = _wrap_deg((curr_peak - base_peak) * sector_width)
                dir_imbalance_delta = float(
                    (np.nanmax(curr_arr[valid_both]) - np.nanmin(curr_arr[valid_both]))
                    - (np.nanmax(base_arr[valid_both]) - np.nanmin(base_arr[valid_both]))
                )
                delta = curr_arr[valid_both] - base_arr[valid_both]
                frac_degraded = float((delta < -6.0).mean())
                frac_improved = float((delta > 6.0).mean())

        baseline_source = det_row.get("baseline_source", "own")

        scores = _score_patterns(
            {**det_row.to_dict(), **scal_row.to_dict()},
            peak_shift_deg,
            dir_imbalance_delta,
            frac_degraded,
            frac_improved,
            baseline_source,
        )

        window = alarms[
            (alarms["cell_id"] == cell_id)
            & (alarms["raised_at"] <= date + pd.Timedelta(days=2))
            & (alarms["cleared_at"] >= date - pd.Timedelta(days=2))
        ]
        matched_alarms = []
        for _, arow in window.iterrows():
            boost = ALARM_BOOST.get(arow["alarm_type"])
            if boost:
                cause, amount = boost
                scores[cause] = float(np.clip(scores.get(cause, 0.0) + amount, 0, 1))
                matched_alarms.append(arow["alarm_type"])

        neigh_list = neighbors.cell_to_neighbors.get(cell_id, [])
        if neigh_list:
            anomalous_neighbors = sum(1 for n in neigh_list if anomaly_lookup.get((n, date), False))
            neighbor_ratio = anomalous_neighbors / len(neigh_list)
        else:
            neighbor_ratio = 0.0
        spatial_scope = "wide-area" if neighbor_ratio >= cfg.neighbor_corr_threshold else "localized"
        if spatial_scope == "wide-area":
            scores = {k: v * 0.7 for k, v in scores.items()}  # de-prioritize per-cell causes

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_cause, top_conf = ranked[0]
        second_cause, second_conf = ranked[1] if len(ranked) > 1 else (None, 0.0)

        rows.append(
            dict(
                cell_id=cell_id,
                date=date,
                top_cause=top_cause,
                top_confidence=round(top_conf, 3),
                second_cause=second_cause,
                second_confidence=round(second_conf, 3),
                recommendation=RECOMMENDATIONS.get(top_cause, ""),
                spatial_scope=spatial_scope,
                neighbor_anomaly_ratio=round(neighbor_ratio, 3),
                matched_alarms=",".join(matched_alarms) if matched_alarms else "",
                baseline_source=baseline_source,
            )
        )

    return pd.DataFrame(rows)
