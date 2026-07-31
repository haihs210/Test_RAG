"""Two-panel visualization for one cell's real Coverage Fingerprint:

  * Panel 1 - the Ring x Direction heatmap (mean RSRP per bin), same
    representation used elsewhere in the pipeline (e.g. Hinh 4 in the
    proposal, ``dashboard.py``'s before/after view).
  * Panel 2 - the raw measurement points themselves: real UE positions
    (relative to the site, in meters) colored by real RSRP, so the binned
    heatmap in panel 1 can be checked against the actual samples it was
    built from.

This is specifically for inspecting real-data extraction (see
``scripts/run_real_data_demo.py``); the synthetic pipeline's alert
dashboard (``dashboard.py``) has its own before/after heatmap for baseline
comparison, which doesn't apply here since a single export is one snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CFSCDConfig
from .features import FingerprintTable


def _ring_direction_matrix(row: pd.Series, cfg: CFSCDConfig, metric: str = "mean") -> np.ndarray:
    n_rings, n_dirs = len(cfg.ring_edges_m), cfg.n_direction_sectors
    mat = np.full((n_rings, n_dirs), np.nan)
    for r in range(n_rings):
        for d in range(n_dirs):
            col = f"rd_{r * n_dirs + d}_{metric}"
            if col in row.index:
                mat[r, d] = row[col]
    return mat


def _ring_labels(cfg: CFSCDConfig) -> list:
    edges = cfg.ring_edges_m
    return [
        f"{int(edges[i])}-{'inf' if i + 1 == len(edges) else int(edges[i + 1])}m" for i in range(len(edges))
    ]


def fig_cell_fingerprint_and_points(
    fp: FingerprintTable,
    geometry_df: pd.DataFrame,
    cell_id: str,
    cfg: CFSCDConfig,
) -> go.Figure:
    """Build the 2-panel figure for one cell.

    ``geometry_df`` must contain, for this cell, ``ue_x_m``/``ue_y_m``
    (absolute UE position), ``site_x_m``/``site_y_m`` (real site position)
    and ``rsrp_dbm`` - i.e. the output of
    ``loader_cell_config.attach_geometry`` / ``attach_bearing``.
    """
    if cell_id not in fp.ring_direction.index.get_level_values("cell_id"):
        raise KeyError(f"{cell_id} has no fingerprint - was it in the UE Report table passed to compute_fingerprints?")
    rd_row = fp.ring_direction.loc[cell_id].iloc[0]
    mat = _ring_direction_matrix(rd_row, cfg)
    ring_labels = _ring_labels(cfg)
    dir_labels = [f"{int(i * 360 / cfg.n_direction_sectors)}" for i in range(cfg.n_direction_sectors)]

    pts = geometry_df[geometry_df["cell_id"] == cell_id].copy()
    pts["dx_m"] = pts["ue_x_m"] - pts["site_x_m"]
    pts["dy_m"] = pts["ue_y_m"] - pts["site_y_m"]

    scalar_row = fp.scalar.loc[cell_id].iloc[0]
    n_samples = int(scalar_row["sample_count"])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Coverage Fingerprint (Ring x Direction) - {cell_id}",
            f"Diem do thuc te (toa do that, n={n_samples})",
        ),
        column_widths=[0.55, 0.45],
    )

    fig.add_trace(
        go.Heatmap(
            z=mat,
            x=dir_labels,
            y=ring_labels,
            colorscale="RdYlGn",
            colorbar=dict(title="RSRP dBm", x=0.44),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=pts["dx_m"],
            y=pts["dy_m"],
            mode="markers",
            marker=dict(
                size=5,
                color=pts["rsrp_dbm"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="RSRP dBm", x=1.0),
                line=dict(width=0),
            ),
            text=[f"{r:.0f} dBm" for r in pts["rsrp_dbm"]],
            hovertemplate="dx=%{x:.0f}m, dy=%{y:.0f}m<br>%{text}<extra></extra>",
            name="mau do",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(size=14, color="black", symbol="star"),
            name="vi tri tram (site)",
            hoverinfo="name",
        ),
        row=1,
        col=2,
    )

    az = geometry_df.loc[geometry_df["cell_id"] == cell_id, "azimuth_deg"]
    if not az.empty and pd.notna(az.iloc[0]) and len(pts):
        az_rad = np.radians(az.iloc[0])
        r = max(float(pts["dx_m"].abs().max()), float(pts["dy_m"].abs().max()), 50.0)
        fig.add_trace(
            go.Scatter(
                x=[0, r * np.sin(az_rad)],
                y=[0, r * np.cos(az_rad)],
                mode="lines",
                line=dict(color="black", dash="dash", width=2),
                name=f"azimuth cau hinh ({az.iloc[0]:.0f} do)",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Huong (do)", row=1, col=1)
    fig.update_yaxes(title_text="Vong ban kinh", row=1, col=1)
    fig.update_xaxes(title_text="Dong Tay (m, tuong doi so voi tram)", row=1, col=2)
    fig.update_yaxes(title_text="Bac Nam (m, tuong doi so voi tram)", row=1, col=2, scaleanchor="x2", scaleratio=1)

    fig.update_layout(
        title=f"Cell {cell_id}: Coverage Fingerprint tu du lieu Mentor thuc te",
        template="plotly_white",
        height=560,
        width=1150,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.75),
        margin=dict(t=90, b=90),
    )
    return fig
