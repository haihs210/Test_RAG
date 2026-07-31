"""Step 1-2 of CF-SCD: Coverage Feature Extraction + Coverage Fingerprint build.

Turns raw UE Report rows (cell_id, date, distance_m, bearing_deg, rsrp_dbm)
into the five feature groups the proposal defines:

  * Signal Feature      - RSRP distribution (mean/median/percentiles, good/poor share)
  * Distance Feature     - effective radius / reach
  * Ring Feature         - RSRP distribution per concentric distance ring
  * Direction Feature    - RSRP distribution per azimuth sector
  * Grid Feature         - RSRP distribution per fixed Cartesian geo-cell

Ring x Direction is additionally kept as a 2D matrix (not just the two 1D
marginals) because that is the "Coverage Fingerprint map" the Spatial
Similarity Index compares cycle-to-cycle (see Hinh 4 in the proposal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CFSCDConfig


@dataclass
class FingerprintTable:
    scalar: pd.DataFrame          # index (cell_id, date) -> Signal + Distance features
    ring: pd.DataFrame            # index (cell_id, date) -> ring_{i}_mean / _density / _count / _confidence
    direction: pd.DataFrame       # index (cell_id, date) -> dir_{i}_mean / _density / _count / _confidence
    grid: pd.DataFrame            # index (cell_id, date) -> grid_{i}_mean / _density / _count / _confidence
    ring_direction: pd.DataFrame  # index (cell_id, date) -> rd_{r*n_dirs+d}_mean / _density / _count / _confidence
    n_rings: int
    n_directions: int
    n_grid_x: int
    n_grid_y: int

    def full_vector(self) -> pd.DataFrame:
        """Concatenate every feature group into one wide fingerprint table."""
        return self.scalar.join([self.ring, self.direction, self.grid], how="outer")


def _bin_ue_reports(ue: pd.DataFrame, cfg: CFSCDConfig) -> pd.DataFrame:
    """Bin each UE Report sample into ring/direction/grid buckets.

    ``bearing_deg`` (and anything derived from it - direction, grid x/y) may
    be missing for some rows: real Mentor exports carry the UE's absolute
    position but not necessarily each site's, so bearing-from-site can only
    be computed where a site position is known (see ``loader_mentor.py``).
    Those rows still contribute to the Signal/Distance/Ring features (which
    only need ``distance_m``); nullable Int64 dtype keeps NaN bearing rows
    as NA bin keys, which groupby then drops rather than crashing on cast.
    """
    ue = ue.copy()
    ring_bins = list(cfg.ring_edges_m) + [np.inf]
    ue["ring"] = pd.cut(
        ue["distance_m"], bins=ring_bins, labels=range(len(ring_bins) - 1), right=False
    ).astype("Int64")

    sector_width = 360.0 / cfg.n_direction_sectors
    ue["direction"] = ((ue["bearing_deg"] // sector_width) % cfg.n_direction_sectors).astype("Int64")

    x_m = ue["distance_m"] * np.sin(np.radians(ue["bearing_deg"]))
    y_m = ue["distance_m"] * np.cos(np.radians(ue["bearing_deg"]))
    n_bins = int(round(2 * cfg.grid_half_extent_m / cfg.grid_cell_m))
    gx = ((x_m + cfg.grid_half_extent_m) // cfg.grid_cell_m).clip(0, n_bins - 1).astype("Int64")
    gy = ((y_m + cfg.grid_half_extent_m) // cfg.grid_cell_m).clip(0, n_bins - 1).astype("Int64")
    ue["grid_x"], ue["grid_y"] = gx, gy
    ue["grid_id"] = gx * n_bins + gy
    ue.attrs["n_grid_bins_per_axis"] = n_bins
    return ue


def _mean_density_table(
    ue: pd.DataFrame, bin_col: str, n_bins: int, prefix: str, confidence_half_count: float
) -> pd.DataFrame:
    """Pivot (cell_id, date, bin) RSRP samples into wide mean/density/count/confidence columns.

    ``confidence`` is a smooth reliability weight in (0, 1], ``count / (count
    + confidence_half_count)`` - it is 0.5 at ``confidence_half_count``
    samples and approaches 1 as sample count grows, so a bin's mean isn't
    trusted the same regardless of how many reports built it (a ring/
    direction/grid cell with 1-2 samples is far noisier than one with 50).
    """
    g = ue.groupby(["cell_id", "date", bin_col], observed=True)["rsrp_dbm"]
    stats = g.agg(mean="mean", count="size").reset_index()
    totals = stats.groupby(["cell_id", "date"])["count"].transform("sum")
    stats["density"] = stats["count"] / totals
    stats["confidence"] = stats["count"] / (stats["count"] + confidence_half_count)

    wide_tables = {}
    for metric, fill in [("mean", np.nan), ("density", 0.0), ("count", 0), ("confidence", 0.0)]:
        wide = stats.pivot_table(index=["cell_id", "date"], columns=bin_col, values=metric, fill_value=fill)
        wide = wide.reindex(columns=range(n_bins), fill_value=fill)
        wide.columns = [f"{prefix}_{i}_{metric}" for i in wide.columns]
        wide_tables[metric] = wide

    return wide_tables["mean"].join([wide_tables["density"], wide_tables["count"], wide_tables["confidence"]])


def compute_fingerprints(ue_reports: pd.DataFrame, cfg: CFSCDConfig) -> FingerprintTable:
    ue = _bin_ue_reports(ue_reports, cfg)
    n_rings = len(cfg.ring_edges_m)
    n_dirs = cfg.n_direction_sectors
    n_grid_axis = int(round(2 * cfg.grid_half_extent_m / cfg.grid_cell_m))
    n_grid = n_grid_axis * n_grid_axis

    scalar = ue.groupby(["cell_id", "date"]).agg(
        sample_count=("rsrp_dbm", "size"),
        rsrp_mean=("rsrp_dbm", "mean"),
        rsrp_median=("rsrp_dbm", "median"),
        rsrp_p10=("rsrp_dbm", lambda s: np.percentile(s, 10)),
        rsrp_p90=("rsrp_dbm", lambda s: np.percentile(s, 90)),
        pct_good=("rsrp_dbm", lambda s: float((s >= cfg.rsrp_good_dbm).mean())),
        pct_poor=("rsrp_dbm", lambda s: float((s <= cfg.rsrp_poor_dbm).mean())),
        dist_mean=("distance_m", "mean"),
        dist_p90=("distance_m", lambda s: np.percentile(s, 90)),  # effective radius
    )

    k = cfg.confidence_half_count
    ring = _mean_density_table(ue, "ring", n_rings, "ring", k)
    direction = _mean_density_table(ue, "direction", n_dirs, "dir", k)
    grid = _mean_density_table(ue, "grid_id", n_grid, "grid", k)

    ue["rd"] = ue["ring"] * n_dirs + ue["direction"]  # both Int64; NA direction -> NA rd, dropped by groupby
    ring_direction = _mean_density_table(ue, "rd", n_rings * n_dirs, "rd", k)

    return FingerprintTable(
        scalar=scalar,
        ring=ring,
        direction=direction,
        grid=grid,
        ring_direction=ring_direction,
        n_rings=n_rings,
        n_directions=n_dirs,
        n_grid_x=n_grid_axis,
        n_grid_y=n_grid_axis,
    )
