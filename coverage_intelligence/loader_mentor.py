"""Loader for real Mentor "power/distance" export files.

This parses the raw event-log export VNPT Mentor actually produces (tab-
separated, one row per measurement event within a call: POWER, DISTANCE,
CQI, PUSCH SINR, THROUGHPUT, ...) into the UE-Report-shaped table the rest
of the pipeline (``features.compute_fingerprints`` etc.) expects.

Two record types matter for the Coverage Fingerprint:

  * ``POWER``    - carries the received signal level (``EC_0``, in dBm - this
                   is the "muc thu" the raw export's serving-cell column) for
                   the serving cell (``ACTIVE_0 == "ACTIVE"``) and up to 11
                   neighbor/candidate cells (``EC_1..EC_11``), plus the UE's
                   absolute position (``X``, ``Y``, in a projected CRS -
                   meters, not lat/lon).
  * ``DISTANCE`` - carries the calibrated distance (meters) from the UE to
                   the serving cell at that moment.

Both are logged independently within the same call (``Call Index``), not at
identical timestamps, so they are joined with a nearest-timestamp match per
(call, cell) rather than a plain merge.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

# "4G-Q01136M12-HCM" -> site "Q01136", sector code "M12", province "HCM"
_NAMED_CELL_RE = re.compile(r"^(?P<tech>[A-Z0-9]+)-(?P<site>[A-Za-z0-9]+?)(?P<sector>[A-Z]\d{2})-(?P<province>.+)$")
# "452-2-6104991-11" -> CGI-style fallback for cells without a friendly name
_CGI_CELL_RE = re.compile(r"^(?P<mcc>\d+)-(?P<mnc>\d+)-(?P<site>\d+)-(?P<sector>\d+)$")


def _derive_site_id(cell_id: str) -> str:
    if not isinstance(cell_id, str):
        return "UNKNOWN"
    m = _NAMED_CELL_RE.match(cell_id)
    if m:
        return f"{m.group('tech')}-{m.group('site')}-{m.group('province')}"
    m = _CGI_CELL_RE.match(cell_id)
    if m:
        return f"{m.group('mcc')}-{m.group('mnc')}-{m.group('site')}"
    return cell_id


def load_mentor_export(path: str, max_join_gap_seconds: float = 30.0) -> pd.DataFrame:
    """Parse a raw Mentor POWER/DISTANCE export into a UE Report table.

    Returns a DataFrame with columns compatible with
    ``features.compute_fingerprints`` (``cell_id``, ``site_id``, ``date``,
    ``distance_m``, ``rsrp_dbm``) plus extras useful for diagnostics
    (``timestamp``, ``ue_x_m``, ``ue_y_m``, ``ecio_db``, ``technology``).

    ``bearing_deg`` is NOT produced here - the export carries the UE's
    absolute position but not each site's, so azimuth-relative-to-site can't
    be computed without a separate cell/site coordinate table. Everything
    that doesn't require bearing (Signal Feature, Distance Feature, and a
    Cartesian Grid Feature built directly from ``ue_x_m``/``ue_y_m``) works
    as-is; see ``docs/algorithm_analysis.md`` update / README for how to add
    site coordinates and unlock the Ring/Direction features too.
    """
    raw = pd.read_csv(path, sep="\t", low_memory=False)

    power = raw[(raw["Record Type"] == "POWER") & (raw["ACTIVE_0"] == "ACTIVE")].copy()
    power = power[["Call Index", "Timestamp", "Technology", "Sector Carrier_0", "EC_0", "EC\\IO_0", "X", "Y"]]
    power.columns = ["call_index", "timestamp", "technology", "cell_id", "rsrp_dbm", "ecio_db", "ue_x_m", "ue_y_m"]
    power["rsrp_dbm"] = pd.to_numeric(power["rsrp_dbm"], errors="coerce")
    power["ecio_db"] = pd.to_numeric(power["ecio_db"], errors="coerce")
    power = power.dropna(subset=["rsrp_dbm"]).sort_values("timestamp")

    dist = raw[(raw["Record Type"] == "DISTANCE") & (raw["ACTIVE_0"] == "ACTIVE")].copy()
    dist = dist[["Call Index", "Timestamp", "Sector Carrier_0", "Calibrated Distance_0"]]
    dist.columns = ["call_index", "timestamp", "cell_id", "distance_m"]
    dist["distance_m"] = pd.to_numeric(dist["distance_m"], errors="coerce")
    dist = dist.dropna(subset=["distance_m"]).sort_values("timestamp")

    merged_parts = []
    for (call_index, cell_id), p_sub in power.groupby(["call_index", "cell_id"]):
        d_sub = dist[(dist["call_index"] == call_index) & (dist["cell_id"] == cell_id)]
        if d_sub.empty:
            continue
        m = pd.merge_asof(
            p_sub.sort_values("timestamp"),
            d_sub[["timestamp", "distance_m"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=int(max_join_gap_seconds * 1000),
        )
        merged_parts.append(m)

    if not merged_parts:
        return pd.DataFrame(
            columns=["cell_id", "site_id", "date", "timestamp", "distance_m", "rsrp_dbm", "ecio_db", "ue_x_m", "ue_y_m", "technology"]
        )

    out = pd.concat(merged_parts, ignore_index=True).dropna(subset=["distance_m"])
    out["site_id"] = out["cell_id"].map(_derive_site_id)
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ms")
    out["date"] = out["timestamp"].dt.floor("D")
    return out[
        ["cell_id", "site_id", "date", "timestamp", "distance_m", "rsrp_dbm", "ecio_db", "ue_x_m", "ue_y_m", "technology"]
    ].reset_index(drop=True)


def estimate_site_positions(ue_reports: pd.DataFrame, max_distance_m: float = 60.0) -> pd.DataFrame:
    """Approximate each cell's site position as the UE position of its
    closest-in observations (small ``distance_m``), since this export does
    not carry site coordinates directly. Cells with no sample inside
    ``max_distance_m`` are omitted - they simply won't get a Direction
    Feature until real site coordinates are supplied.
    """
    near = ue_reports[ue_reports["distance_m"] <= max_distance_m]
    if near.empty:
        return pd.DataFrame(columns=["cell_id", "site_x_m", "site_y_m", "n_near_samples"])
    return (
        near.groupby("cell_id")
        .agg(site_x_m=("ue_x_m", "median"), site_y_m=("ue_y_m", "median"), n_near_samples=("ue_x_m", "size"))
        .reset_index()
    )
