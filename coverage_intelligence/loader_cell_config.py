"""Loader for a real cell/site RF & installation config export (e.g. from RIMS).

This is the piece that was missing from ``loader_mentor.py`` alone: the raw
Mentor UE Report export carries the UE's position but not each site's, so
Direction/Ring-x-Direction (which need bearing-from-site) could only be
approximated. A cell config export like the one this module reads - real
Latitude/Longitude, azimuth, mechanical/electrical tilt, antenna height and
gain per cell - lets bearing be computed exactly instead of estimated.

Expected columns (Vietnamese headers, as exported): ``Tên trên hệ thống``
(cell name), ``Longtitude``, ``Latitude``, ``pci``, ``tac``, ``lcrid``,
``Băng tần`` (band), ``Antenna gain``, ``Antenna high``, ``Mechaincal
tilt``, ``Electrical tilt``, ``Total tilt``, ``azimuth``, ``Trạng thái hoạt
động`` (operational status: Onair/Offair).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer

from .loader_mentor import _derive_site_id, estimate_site_positions

# Vietnam sits entirely in UTM zone 48N; this matches the projected CRS the
# Mentor export's own X/Y columns use (verified empirically: converting a
# cell's Lat/Long here lands within ~100m of the UE positions Mentor logged
# while being served by that same cell).
_LATLON_TO_UTM48N = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)

_COLUMN_MAP = {
    "Tên trên hệ thống": "cell_id",
    "Longtitude": "lon",
    "Latitude": "lat",
    "pci": "pci",
    "tac": "tac",
    "lcrid": "lcrid",
    "Băng tần": "band",
    "Antenna gain": "antenna_gain",
    "Antenna high": "antenna_height_m",
    "Mechaincal tilt": "mech_tilt_deg",
    "Electrical tilt": "elec_tilt_deg",
    "Total tilt": "total_tilt_deg",
    "azimuth": "azimuth_deg",
    "Trạng thái hoạt động": "status",
}


def load_cell_config_xlsx(path: str) -> pd.DataFrame:
    """Parse a cell config export into a table with real site coordinates
    (projected to meters, ``site_x_m``/``site_y_m``) alongside the raw
    lat/lon and RF/installation attributes.
    """
    raw = pd.read_excel(path)
    missing = set(_COLUMN_MAP) - set(raw.columns)
    if missing:
        raise ValueError(f"cell config file is missing expected columns: {sorted(missing)}")

    df = raw[list(_COLUMN_MAP)].rename(columns=_COLUMN_MAP)
    df["site_id"] = df["cell_id"].map(_derive_site_id)

    has_coords = df["lon"].notna() & df["lat"].notna()
    x, y = np.full(len(df), np.nan), np.full(len(df), np.nan)
    if has_coords.any():
        xs, ys = _LATLON_TO_UTM48N.transform(
            df.loc[has_coords, "lon"].to_numpy(), df.loc[has_coords, "lat"].to_numpy()
        )
        x[has_coords.to_numpy()] = xs
        y[has_coords.to_numpy()] = ys
    df["site_x_m"] = x
    df["site_y_m"] = y

    return df


def attach_real_bearing(ue_reports: pd.DataFrame, cell_config: pd.DataFrame) -> pd.DataFrame:
    """Join real site coordinates onto UE Report rows (by exact ``cell_id``
    match) and compute the true bearing from site to UE. Rows for cells
    absent from ``cell_config`` are left with ``bearing_deg`` unset - callers
    can layer ``loader_mentor.estimate_site_positions`` on top as a fallback
    for those.
    """
    coords = cell_config.dropna(subset=["site_x_m", "site_y_m"])[
        ["cell_id", "site_x_m", "site_y_m", "azimuth_deg", "mech_tilt_deg", "elec_tilt_deg", "band", "status"]
    ].drop_duplicates(subset="cell_id")

    out = ue_reports.merge(coords, on="cell_id", how="left")
    dx = out["ue_x_m"] - out["site_x_m"]
    dy = out["ue_y_m"] - out["site_y_m"]
    out["bearing_deg"] = np.degrees(np.arctan2(dx, dy)) % 360
    return out


def attach_bearing(
    ue_reports: pd.DataFrame,
    cell_config: pd.DataFrame,
    fallback_max_distance_m: float = 60.0,
) -> pd.DataFrame:
    """``attach_real_bearing`` for cells present in ``cell_config``, falling
    back to ``loader_mentor.estimate_site_positions`` (closest-in-sample
    proxy) for the rest, so as many rows as possible get a usable bearing.
    """
    out = attach_real_bearing(ue_reports, cell_config)
    missing_mask = out["bearing_deg"].isna()
    if not missing_mask.any():
        return out

    missing = out.loc[missing_mask].drop(
        columns=["bearing_deg", "site_x_m", "site_y_m", "azimuth_deg", "mech_tilt_deg", "elec_tilt_deg", "band", "status"],
        errors="ignore",
    )
    est_sites = estimate_site_positions(missing, max_distance_m=fallback_max_distance_m)
    missing = missing.merge(est_sites[["cell_id", "site_x_m", "site_y_m"]], on="cell_id", how="left")
    dx = missing["ue_x_m"] - missing["site_x_m"]
    dy = missing["ue_y_m"] - missing["site_y_m"]
    est_bearing = np.degrees(np.arctan2(dx, dy)) % 360

    out.loc[missing_mask, "bearing_deg"] = est_bearing.to_numpy()
    return out
