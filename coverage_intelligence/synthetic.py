"""Large-scale synthetic data generator for the CF-SCD pipeline.

The real system ingests UE Report from the Mentor database, RF/installation
data from resource-management systems (e.g. RIMS) and Alarm from nFM (see
``docs/algorithm_analysis.md`` / the README for how a production loader would
plug into the same pipeline). None of that is reachable from here, so this
module produces statistically realistic UE Report / cell-config / alarm data
at whatever scale is needed (hundreds to millions of rows) using a simple
log-distance propagation model with antenna radiation patterns, so the rest
of the pipeline can be built, tested and demonstrated end-to-end.

Five named anomaly scenarios cover the algorithm's typical failure modes:
OUTAGE, SHADOWING, TILT_POWER_DRIFT, AZIMUTH_DRIFT, OVERSHOOT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0

BANDS = [
    # (name, tech, typical design radius in meters, path-loss exponent)
    ("B900", "4G", 2200.0, 3.3),
    ("B1800", "4G", 1400.0, 3.5),
    ("B2600", "4G", 900.0, 3.7),
    ("N78", "5G", 700.0, 3.8),
]

SCENARIO_TYPES = [
    "OUTAGE",
    "SHADOWING",
    "TILT_POWER_DRIFT",
    "AZIMUTH_DRIFT",
    "OVERSHOOT",
    "NONE",
]


def _wrap_deg(delta: np.ndarray) -> np.ndarray:
    """Wrap an angle difference (degrees) into [-180, 180]."""
    return (delta + 180.0) % 360.0 - 180.0


@dataclass
class ScenarioSpec:
    cell_id: str
    kind: str
    start_day: int
    params: Dict


def make_cell_topology(
    n_cells: int,
    rng: np.random.Generator,
    n_sites: Optional[int] = None,
    area_km: float = 30.0,
    cells_per_site: int = 3,
) -> pd.DataFrame:
    """Create a synthetic multi-vendor, multi-band cell topology.

    Sites are scattered over a square area; each site carries 1-3 sectors
    (cells) at randomized azimuths. Returned columns mirror standard RF/
    install attributes (Cell ID, Site ID, band, PCI, ARFCN, Azimuth,
    Mechanical Tilt, Electrical Tilt, antenna height, tx power, coordinates).
    """
    if n_sites is None:
        n_sites = max(1, n_cells // cells_per_site)

    site_x = rng.uniform(-area_km * 500, area_km * 500, size=n_sites)
    site_y = rng.uniform(-area_km * 500, area_km * 500, size=n_sites)
    vendors = rng.choice(["Ericsson", "Nokia", "Huawei", "ZTE"], size=n_sites)
    provinces = rng.choice(
        ["Ha Noi", "TP.HCM", "Da Nang", "Hai Phong", "Can Tho"], size=n_sites
    )

    rows = []
    cell_counter = 0
    site_idx = 0
    while cell_counter < n_cells:
        n_sectors = int(rng.choice([1, 2, 3], p=[0.1, 0.2, 0.7]))
        base_azimuths = np.linspace(0, 360, n_sectors, endpoint=False)
        base_azimuths = (base_azimuths + rng.uniform(0, 30)) % 360
        for k in range(n_sectors):
            if cell_counter >= n_cells:
                break
            band, tech, design_radius, plexp = BANDS[rng.integers(0, len(BANDS))]
            cell_counter += 1
            rows.append(
                dict(
                    cell_id=f"CELL{cell_counter:06d}",
                    site_id=f"SITE{site_idx % n_sites:05d}",
                    vendor=vendors[site_idx % n_sites],
                    province=provinces[site_idx % n_sites],
                    band=band,
                    tech=tech,
                    pci=int(rng.integers(0, 504)),
                    arfcn=int(rng.integers(1000, 60000)),
                    azimuth_deg=float(base_azimuths[k]),
                    mech_tilt_deg=float(rng.uniform(0, 4)),
                    elec_tilt_deg=float(rng.uniform(2, 10)),
                    antenna_height_m=float(rng.uniform(25, 45)),
                    tx_power_dbm=float(rng.uniform(43, 49)),
                    beamwidth_h_deg=float(rng.choice([65.0, 90.0])),
                    beamwidth_v_deg=float(rng.uniform(6, 10)),
                    design_radius_m=design_radius,
                    path_loss_exponent=plexp,
                    x_m=float(site_x[site_idx % n_sites]),
                    y_m=float(site_y[site_idx % n_sites]),
                )
            )
        site_idx += 1

    cells = pd.DataFrame(rows)
    cells["peer_group"] = cells["band"] + "_" + cells["tech"]
    return cells


def pick_scenarios(
    cells: pd.DataFrame,
    rng: np.random.Generator,
    n_per_type: int = 3,
    start_day_range=(18, 24),
) -> List[ScenarioSpec]:
    """Deterministically inject a handful of each anomaly type for the demo."""
    specs: List[ScenarioSpec] = []
    pool = cells["cell_id"].tolist()
    rng.shuffle(pool)
    idx = 0
    for kind in ["OUTAGE", "SHADOWING", "TILT_POWER_DRIFT", "AZIMUTH_DRIFT"]:
        for _ in range(n_per_type):
            if idx >= len(pool):
                break
            start_day = int(rng.integers(*start_day_range))
            params = {}
            if kind == "SHADOWING":
                params = dict(center_bearing=float(rng.uniform(0, 360)), extra_loss_db=14.0)
            elif kind == "TILT_POWER_DRIFT":
                params = dict(
                    extra_tilt_deg=float(rng.uniform(3, 7)),
                    power_drop_db=float(rng.uniform(2, 6)),
                )
            elif kind == "AZIMUTH_DRIFT":
                params = dict(drift_deg=float(rng.choice([-1, 1])) * rng.uniform(25, 60))
            specs.append(ScenarioSpec(pool[idx], kind, start_day, params))
            idx += 1

    # Overshoot: pick "new" cells that only get a short history so RCA must
    # fall back to peer-group comparison instead of the cell's own baseline.
    for _ in range(n_per_type):
        if idx >= len(pool):
            break
        specs.append(
            ScenarioSpec(
                pool[idx],
                "OVERSHOOT",
                start_day=0,
                params=dict(extra_power_db=float(rng.uniform(6, 10)), uptilt_deg=float(rng.uniform(3, 6))),
            )
        )
        idx += 1
    return specs


def _propagation_rsrp(
    distance_m: np.ndarray,
    bearing_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    tilt_deg: np.ndarray,
    height_m: np.ndarray,
    tx_power_dbm: np.ndarray,
    plexp: np.ndarray,
    beamwidth_h: np.ndarray,
    beamwidth_v: np.ndarray,
    rng: np.random.Generator,
    shadow_sigma_db: float = 6.0,
) -> np.ndarray:
    d = np.clip(distance_m, 10.0, None)
    d0, pl0 = 100.0, 100.0
    path_loss = pl0 + 10 * plexp * np.log10(d / d0)

    h_delta = _wrap_deg(bearing_deg - azimuth_deg)
    h_loss = np.minimum(12.0 * (h_delta / beamwidth_h) ** 2, 25.0)

    elevation_deg = np.degrees(np.arctan2(np.maximum(height_m - 1.5, 1.0), d))
    v_delta = elevation_deg - tilt_deg
    v_loss = np.minimum(12.0 * (v_delta / beamwidth_v) ** 2, 20.0)

    shadow_fading = rng.normal(0.0, shadow_sigma_db, size=d.shape[0])
    rsrp = tx_power_dbm - path_loss - h_loss - v_loss + shadow_fading
    return np.clip(rsrp, -140.0, -44.0)


def generate_ue_reports(
    cells: pd.DataFrame,
    scenarios: List[ScenarioSpec],
    start_date: str,
    n_days: int,
    base_samples_per_cell_day: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Vectorized generation of the UE Report fact table.

    Runtime and memory scale with total row count (n_cells * n_days *
    base_samples_per_cell_day), which is computed up front and expanded with
    ``np.repeat`` rather than a per-sample Python loop, so multi-million row
    datasets are practical on a laptop.
    """
    scen_by_cell = {s.cell_id: s for s in scenarios}
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    n_cells = len(cells)
    cell_idx_arr = np.arange(n_cells)
    day_idx_arr = np.arange(n_days)
    grid_cell, grid_day = np.meshgrid(cell_idx_arr, day_idx_arr, indexing="ij")
    grid_cell = grid_cell.ravel()
    grid_day = grid_day.ravel()

    lam = np.full(grid_cell.shape[0], float(base_samples_per_cell_day))
    for cid, spec in scen_by_cell.items():
        cpos = cells.index[cells["cell_id"] == cid]
        if len(cpos) == 0:
            continue
        ci = cpos[0]
        mask = (grid_cell == ci) & (grid_day >= spec.start_day)
        if spec.kind == "OUTAGE":
            lam[mask] *= 0.03
        elif spec.kind == "OVERSHOOT":
            # only a handful of days of history before "commissioning"
            history_mask = (grid_cell == ci) & (grid_day < 3)
            lam[(grid_cell == ci)] = 0.0
            lam[history_mask] = base_samples_per_cell_day

    lam = np.maximum(lam, 0.0)
    sample_counts = rng.poisson(lam)
    total_rows = int(sample_counts.sum())

    row_to_flat = np.repeat(np.arange(grid_cell.shape[0]), sample_counts)
    row_cell_idx = grid_cell[row_to_flat]
    row_day_idx = grid_day[row_to_flat]

    c = cells.iloc[row_cell_idx].reset_index(drop=True)
    day_of = row_day_idx

    design_radius = c["design_radius_m"].to_numpy().copy()
    tx_power = c["tx_power_dbm"].to_numpy().copy()
    azimuth = c["azimuth_deg"].to_numpy().copy()
    tilt_total = (c["mech_tilt_deg"] + c["elec_tilt_deg"]).to_numpy().copy()

    bearing = rng.uniform(0, 360, size=total_rows)
    radius_scale = design_radius / 1.4
    distance = rng.gamma(shape=1.6, scale=1.0, size=total_rows) * radius_scale

    extra_shadow_loss = np.zeros(total_rows)

    for cid, spec in scen_by_cell.items():
        cpos = cells.index[cells["cell_id"] == cid]
        if len(cpos) == 0:
            continue
        ci = cpos[0]
        m = (row_cell_idx == ci) & (day_of >= spec.start_day)
        if not m.any():
            continue
        if spec.kind == "OUTAGE":
            extra_shadow_loss[m] += 40.0
        elif spec.kind == "SHADOWING":
            days_since = (day_of[m] - spec.start_day).astype(float)
            half_width = 20.0 + np.minimum(days_since, 10.0) * 4.0  # widens over time
            delta = np.abs(_wrap_deg(bearing[m] - spec.params["center_bearing"]))
            in_shadow = delta < half_width
            loss = np.zeros(m.sum())
            loss[in_shadow] = spec.params["extra_loss_db"]
            extra_shadow_loss[m] += loss
        elif spec.kind == "TILT_POWER_DRIFT":
            tilt_total[m] += spec.params["extra_tilt_deg"]
            tx_power[m] -= spec.params["power_drop_db"]
            distance[m] *= 0.75  # downtilt shrinks the effective footprint
        elif spec.kind == "AZIMUTH_DRIFT":
            azimuth[m] = (azimuth[m] + spec.params["drift_deg"]) % 360
        elif spec.kind == "OVERSHOOT":
            tx_power[m] += spec.params["extra_power_db"]
            tilt_total[m] = np.maximum(tilt_total[m] - spec.params["uptilt_deg"], 0.5)
            distance[m] *= 1.9

    rsrp = _propagation_rsrp(
        distance_m=distance,
        bearing_deg=bearing,
        azimuth_deg=azimuth,
        tilt_deg=tilt_total,
        height_m=c["antenna_height_m"].to_numpy(),
        tx_power_dbm=tx_power,
        plexp=c["path_loss_exponent"].to_numpy(),
        beamwidth_h=c["beamwidth_h_deg"].to_numpy(),
        beamwidth_v=c["beamwidth_v_deg"].to_numpy(),
        rng=rng,
    )
    rsrp -= extra_shadow_loss

    df = pd.DataFrame(
        {
            "cell_id": c["cell_id"].to_numpy(),
            "site_id": c["site_id"].to_numpy(),
            "date": dates.values[day_of],
            "distance_m": distance,
            "bearing_deg": bearing,
            "rsrp_dbm": np.clip(rsrp, -140.0, -44.0),
        }
    )
    return df


def generate_alarms(
    cells: pd.DataFrame,
    scenarios: List[ScenarioSpec],
    start_date: str,
    n_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Synthetic Alarm feed (queried, in production, from nFM via API)."""
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    alarm_map = {
        "OUTAGE": ("CELL_OUTAGE", "Critical"),
        "TILT_POWER_DRIFT": ("TX_POWER_LOW", "Major"),
        "SHADOWING": None,  # environmental, no equipment alarm expected
        "AZIMUTH_DRIFT": None,  # mechanical drift, no equipment alarm expected
        "OVERSHOOT": None,
    }
    rows = []
    aid = 1
    for spec in scenarios:
        mapping = alarm_map.get(spec.kind)
        if mapping is None:
            continue
        alarm_type, severity = mapping
        raised = dates[min(spec.start_day, n_days - 1)]
        cleared = dates[min(spec.start_day + int(rng.integers(1, 4)), n_days - 1)]
        rows.append(
            dict(
                alarm_id=f"ALM{aid:06d}",
                cell_id=spec.cell_id,
                alarm_type=alarm_type,
                severity=severity,
                raised_at=raised,
                cleared_at=cleared,
            )
        )
        aid += 1
    return pd.DataFrame(rows, columns=["alarm_id", "cell_id", "alarm_type", "severity", "raised_at", "cleared_at"])


def generate_demo_dataset(
    n_cells: int = 300,
    n_days: int = 35,
    base_samples_per_cell_day: int = 800,
    start_date: str = "2026-06-01",
    seed: int = 42,
):
    """Convenience wrapper: topology + scenarios + UE reports + alarms."""
    rng = np.random.default_rng(seed)
    cells = make_cell_topology(n_cells, rng)
    scenarios = pick_scenarios(cells, rng)
    ue_reports = generate_ue_reports(cells, scenarios, start_date, n_days, base_samples_per_cell_day, rng)
    alarms = generate_alarms(cells, scenarios, start_date, n_days, rng)
    return cells, scenarios, ue_reports, alarms
