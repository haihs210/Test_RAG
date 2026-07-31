"""Step 3 of CF-SCD: Baseline Construction.

Per-cell rolling baseline (mean / median / MAD) over a 7-30 day trailing
window (``cfg.baseline_window_days``), with confirmed-incident days and
statistical outlier days excluded: the baseline is built from each cell's
own trailing history, excluding confirmed-incident days and outliers, using
mean, median and - notably - Median Absolute Deviation (MAD) together.

The Coverage Fingerprint table is already one row per (cell, day) - a few
thousand to tens of thousands of rows even when the raw UE Report behind it
is in the hundreds of millions - so this operates comfortably in memory.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

from .config import CFSCDConfig


@dataclass
class BaselineResult:
    median: pd.DataFrame
    mad: pd.DataFrame
    is_valid: pd.Series  # True where >= min_baseline_days of clean history exist


def find_outlier_dates(
    scalar: pd.DataFrame,
    cfg: CFSCDConfig,
    confirmed_incident_dates: Optional[Dict[str, Set]] = None,
    key_column: str = "rsrp_mean",
    z_threshold: float = 3.0,
) -> Dict[str, Set]:
    """Flag statistically outlying days (per cell) on the key scalar metric.

    These, plus any human-confirmed incident days, are excluded when the
    trailing-window baseline is built below.
    """
    confirmed_incident_dates = confirmed_incident_dates or {}
    excluded: Dict[str, Set] = {cid: set(dates) for cid, dates in confirmed_incident_dates.items()}

    for cell_id, sub in scalar[[key_column]].groupby(level=0):
        sub = sub.droplevel(0).sort_index()
        values = sub[key_column].to_numpy()
        dates = sub.index.to_numpy()
        cell_excluded = excluded.setdefault(cell_id, set())
        for i in range(len(dates)):
            start = max(0, i - cfg.baseline_window_days)
            window = values[start:i]
            if window.shape[0] < cfg.min_baseline_days:
                continue
            med = np.nanmedian(window)
            mad = np.nanmedian(np.abs(window - med)) * cfg.mad_scale
            mad = max(mad, 1e-6)
            z = abs(values[i] - med) / mad
            if z > z_threshold:
                cell_excluded.add(dates[i])
    return excluded


def compute_baseline(
    wide: pd.DataFrame,
    cfg: CFSCDConfig,
    excluded_dates: Optional[Dict[str, Set]] = None,
) -> BaselineResult:
    """Trailing-window robust baseline for every column of a wide fingerprint table.

    ``wide`` must be indexed by (cell_id, date) with date sorted ascending
    within each cell (the caller-visible ``full_vector`` / per-group tables
    from ``features.py`` satisfy this once sorted).
    """
    excluded_dates = excluded_dates or {}
    wide = wide.sort_index()
    median_out = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    mad_out = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    valid_out = pd.Series(False, index=wide.index)

    for cell_id, sub in wide.groupby(level=0):
        sub = sub.droplevel(0).sort_index()
        dates = sub.index.to_numpy()
        values = sub.to_numpy(dtype=float)
        cell_excluded = excluded_dates.get(cell_id, set())
        excl_mask = np.array([d in cell_excluded for d in dates])

        for i in range(len(dates)):
            start = max(0, i - cfg.baseline_window_days)
            idx = np.arange(start, i)
            idx = idx[~excl_mask[idx]]
            row_key = (cell_id, dates[i])
            if idx.shape[0] < cfg.min_baseline_days:
                continue
            window = values[idx]
            with warnings.catch_warnings():
                # sparse ring/direction/grid bins can be all-NaN in a given
                # window (no UE Report sample ever landed there) - expected.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(window, axis=0)
                mad = np.nanmedian(np.abs(window - med), axis=0) * cfg.mad_scale
            median_out.loc[row_key] = med
            mad_out.loc[row_key] = mad
            valid_out.loc[row_key] = True

    return BaselineResult(median=median_out, mad=mad_out, is_valid=valid_out)


def peer_group_baseline(
    wide: pd.DataFrame,
    cell_to_peer_group: pd.Series,
    cfg: CFSCDConfig,
) -> BaselineResult:
    """Cross-sectional baseline built from peer cells (same band/tech) on the
    same day, used for cells whose own history is too short (e.g. newly
    commissioned sites) - see the "Overshoot" scenario, which needs an
    accurate evaluation even for a new site with no history of its own yet.
    """
    df = wide.join(cell_to_peer_group.rename("peer_group"))
    median_out = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    mad_out = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    valid_out = pd.Series(False, index=wide.index)

    for (peer_group, date), sub in df.groupby(["peer_group", df.index.get_level_values("date")]):
        feature_cols = wide.columns
        values = sub[feature_cols].to_numpy(dtype=float)
        if values.shape[0] < cfg.min_baseline_days:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(values, axis=0)
            mad = np.nanmedian(np.abs(values - med), axis=0) * cfg.mad_scale
        for cell_id in sub.index.get_level_values("cell_id"):
            row_key = (cell_id, date)
            median_out.loc[row_key] = med
            mad_out.loc[row_key] = mad
            valid_out.loc[row_key] = True

    return BaselineResult(median=median_out, mad=mad_out, is_valid=valid_out)
