"""Three-panel visualization for one cell's real Coverage Fingerprint:

  * Panel 1 - Ring x Direction heatmap (mean RSRP per bin), same
    representation used elsewhere in the pipeline (e.g. Hinh 4 in the
    proposal, ``dashboard.py``'s before/after view). Hover shows each bin's
    sample count and confidence.
  * Panel 2 - "coverage rose": the same Ring x Direction data redrawn as
    real angular wedges (``go.Barpolar``) radiating from the site, so the
    shape actually looks like antenna sector coverage instead of an
    abstract ring-index x direction-index grid. Each wedge's opacity is
    weighted by its confidence (sample count), so sparsely-sampled bins
    fade out instead of implying certainty they don't have.
  * Panel 3 - the raw measurement points themselves: real UE positions
    (relative to the site, in meters) colored by real RSRP, so panels 1-2
    can be checked against the actual samples they were built from.

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


def _ring_display_edges(cfg: CFSCDConfig) -> list:
    """Outer edge for every ring, giving the last (open-ended) ring a finite
    display width equal to the previous ring's width, purely for plotting.
    """
    edges = list(cfg.ring_edges_m)
    last_width = edges[-1] - edges[-2] if len(edges) > 1 else edges[-1]
    return edges + [edges[-1] + last_width]


def _add_ring_direction_heatmap(fig: go.Figure, rd_row: pd.Series, cfg: CFSCDConfig, row: int, col: int, coloraxis: str):
    mean_mat = _ring_direction_matrix(rd_row, cfg, "mean")
    count_mat = _ring_direction_matrix(rd_row, cfg, "count")
    conf_mat = _ring_direction_matrix(rd_row, cfg, "confidence")
    ring_labels = _ring_labels(cfg)
    dir_labels = [f"{int(i * 360 / cfg.n_direction_sectors)}" for i in range(cfg.n_direction_sectors)]

    customdata = np.dstack([count_mat, conf_mat])
    fig.add_trace(
        go.Heatmap(
            z=mean_mat,
            x=dir_labels,
            y=ring_labels,
            coloraxis=coloraxis,
            customdata=customdata,
            hovertemplate=(
                "huong=%{x} do, vong=%{y}<br>RSRP=%{z:.1f} dBm"
                "<br>so mau (n)=%{customdata[0]:.0f}, do tin cay=%{customdata[1]:.2f}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )


def _add_coverage_rose(fig: go.Figure, rd_row: pd.Series, cfg: CFSCDConfig, row: int, col: int, coloraxis: str):
    """Draw each ring as a stacked go.Barpolar layer covering all sectors -
    together the layers form true annular wedges (sector shape, vertex at
    the site), rather than the rectangular index-based heatmap.
    """
    n_rings, n_dirs = len(cfg.ring_edges_m), cfg.n_direction_sectors
    sector_width = 360.0 / n_dirs
    theta_centers = [i * sector_width + sector_width / 2 for i in range(n_dirs)]
    display_edges = _ring_display_edges(cfg)

    for r in range(n_rings):
        means, counts, confs = [], [], []
        for d in range(n_dirs):
            means.append(rd_row.get(f"rd_{r * n_dirs + d}_mean", np.nan))
            counts.append(rd_row.get(f"rd_{r * n_dirs + d}_count", 0.0))
            confs.append(rd_row.get(f"rd_{r * n_dirs + d}_confidence", 0.0))
        means = np.array(means, dtype=float)
        # invisible where there is literally no sample, rather than a
        # misleadingly-solid wedge with an undefined color
        opacity = [0.0 if np.isnan(m) else max(float(c), 0.08) for m, c in zip(means, confs)]

        fig.add_trace(
            go.Barpolar(
                r=[display_edges[r + 1] - display_edges[r]] * n_dirs,
                base=[display_edges[r]] * n_dirs,
                theta=theta_centers,
                width=[sector_width * 0.92] * n_dirs,
                marker=dict(
                    color=means,
                    coloraxis=coloraxis,
                    opacity=opacity,
                    line=dict(width=0.5, color="white"),
                ),
                customdata=np.stack([counts, confs], axis=-1),
                hovertemplate=(
                    "vong %{base:.0f}-%{r:.0f}m them, huong~%{theta:.0f} do<br>"
                    "so mau (n)=%{customdata[0]:.0f}, do tin cay=%{customdata[1]:.2f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def build_fingerprint_report(fp: FingerprintTable, cell_id: str, cfg: CFSCDConfig) -> str:
    """Render every Coverage Fingerprint feature group the proposal defines
    - Signal, Distance, Ring, Direction, Grid - as one readable text report
    for a single cell, rather than just the scalar (Signal+Distance) table
    the top-N printout already shows.
    """
    scalar_row = fp.scalar.loc[cell_id].iloc[0]
    lines = [f"=== Coverage Fingerprint report: {cell_id} ===", ""]

    lines.append("-- Signal Feature --")
    lines.append(f"  sample_count = {int(scalar_row['sample_count'])}")
    lines.append(f"  rsrp_mean = {scalar_row['rsrp_mean']:.1f} dBm, rsrp_median = {scalar_row['rsrp_median']:.1f} dBm")
    lines.append(f"  rsrp_p10 = {scalar_row['rsrp_p10']:.1f} dBm, rsrp_p90 = {scalar_row['rsrp_p90']:.1f} dBm")
    lines.append(f"  pct_good (>= {cfg.rsrp_good_dbm:.0f} dBm) = {scalar_row['pct_good'] * 100:.1f}%, "
                 f"pct_poor (<= {cfg.rsrp_poor_dbm:.0f} dBm) = {scalar_row['pct_poor'] * 100:.1f}%")
    lines.append("")

    lines.append("-- Distance Feature --")
    lines.append(f"  dist_mean = {scalar_row['dist_mean']:.0f} m")
    lines.append(f"  dist_p90 (ban kinh hieu dung / effective radius) = {scalar_row['dist_p90']:.0f} m")
    lines.append("")

    lines.append("-- Ring Feature (marginal, gop tat ca huong) --")
    ring_row = fp.ring.loc[cell_id].iloc[0]
    ring_labels = _ring_labels(cfg)
    for i, label in enumerate(ring_labels):
        mean = ring_row.get(f"ring_{i}_mean", np.nan)
        count = ring_row.get(f"ring_{i}_count", 0)
        conf = ring_row.get(f"ring_{i}_confidence", 0.0)
        if count > 0:
            lines.append(f"  {label:>12}: RSRP={mean:7.1f} dBm  n={int(count):4d}  do_tin_cay={conf:.2f}")
    lines.append("")

    lines.append("-- Direction Feature (marginal, gop tat ca vong) --")
    dir_row = fp.direction.loc[cell_id].iloc[0]
    for i in range(cfg.n_direction_sectors):
        mean = dir_row.get(f"dir_{i}_mean", np.nan)
        count = dir_row.get(f"dir_{i}_count", 0)
        conf = dir_row.get(f"dir_{i}_confidence", 0.0)
        deg = int(i * 360 / cfg.n_direction_sectors)
        if count > 0:
            lines.append(f"  {deg:3d} do: RSRP={mean:7.1f} dBm  n={int(count):4d}  do_tin_cay={conf:.2f}")
    lines.append("")

    lines.append("-- Grid Feature (o luoi Descartes co dinh, chi liet ke o co du lieu) --")
    grid_row = fp.grid.loc[cell_id].iloc[0]
    n_grid = fp.n_grid_x * fp.n_grid_y
    grid_hits = []
    for i in range(n_grid):
        count = grid_row.get(f"grid_{i}_count", 0)
        if count > 0:
            grid_hits.append((i, grid_row.get(f"grid_{i}_mean", np.nan), int(count), grid_row.get(f"grid_{i}_confidence", 0.0)))
    grid_hits.sort(key=lambda t: t[2], reverse=True)
    lines.append(f"  {len(grid_hits)}/{n_grid} o luoi co du lieu")
    for i, mean, count, conf in grid_hits[:10]:
        lines.append(f"    grid_{i}: RSRP={mean:7.1f} dBm  n={count:4d}  do_tin_cay={conf:.2f}")

    return "\n".join(lines)


def fig_cell_coverage_report(
    fp: FingerprintTable,
    geometry_df: pd.DataFrame,
    cell_id: str,
    cfg: CFSCDConfig,
) -> go.Figure:
    """Build the 3-panel figure for one cell: Ring x Direction heatmap,
    polar coverage rose (sector wedges, confidence-weighted opacity), and
    the real measurement points.

    ``geometry_df`` must contain, for this cell, ``ue_x_m``/``ue_y_m``
    (absolute UE position), ``site_x_m``/``site_y_m`` (real site position)
    and ``rsrp_dbm`` - i.e. the output of
    ``loader_cell_config.attach_geometry`` / ``attach_bearing``.
    """
    if cell_id not in fp.ring_direction.index.get_level_values("cell_id"):
        raise KeyError(f"{cell_id} has no fingerprint - was it in the UE Report table passed to compute_fingerprints?")
    rd_row = fp.ring_direction.loc[cell_id].iloc[0]
    scalar_row = fp.scalar.loc[cell_id].iloc[0]
    n_samples = int(scalar_row["sample_count"])

    pts = geometry_df[geometry_df["cell_id"] == cell_id].copy()
    pts["dx_m"] = pts["ue_x_m"] - pts["site_x_m"]
    pts["dy_m"] = pts["ue_y_m"] - pts["site_y_m"]

    mean_mat = _ring_direction_matrix(rd_row, cfg, "mean")
    all_rsrp = np.concatenate([mean_mat[~np.isnan(mean_mat)].ravel(), pts["rsrp_dbm"].dropna().to_numpy()])
    cmin, cmax = (float(np.min(all_rsrp)), float(np.max(all_rsrp))) if all_rsrp.size else (-120.0, -60.0)

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "xy"}, {"type": "polar"}, {"type": "xy"}]],
        subplot_titles=(
            "Ring x Direction (luoi)",
            "Coverage rose (dang canh quat, do mo = do tin cay)",
            f"Diem do thuc te (n={n_samples})",
        ),
        column_widths=[0.34, 0.34, 0.32],
    )

    _add_ring_direction_heatmap(fig, rd_row, cfg, row=1, col=1, coloraxis="coloraxis")
    _add_coverage_rose(fig, rd_row, cfg, row=1, col=2, coloraxis="coloraxis")

    fig.add_trace(
        go.Scatter(
            x=pts["dx_m"],
            y=pts["dy_m"],
            mode="markers",
            marker=dict(size=5, color=pts["rsrp_dbm"], coloraxis="coloraxis", line=dict(width=0)),
            text=[f"{r:.0f} dBm" for r in pts["rsrp_dbm"]],
            hovertemplate="dx=%{x:.0f}m, dy=%{y:.0f}m<br>%{text}<extra></extra>",
            name="mau do",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0], mode="markers", marker=dict(size=14, color="black", symbol="star"),
            name="vi tri tram (site)", hoverinfo="name",
        ),
        row=1,
        col=3,
    )

    az = geometry_df.loc[geometry_df["cell_id"] == cell_id, "azimuth_deg"]
    if not az.empty and pd.notna(az.iloc[0]):
        az_val = float(az.iloc[0])
        r_line = max(float(pts["dx_m"].abs().max()) if len(pts) else 0.0, float(pts["dy_m"].abs().max()) if len(pts) else 0.0, 50.0)
        az_rad = np.radians(az_val)
        fig.add_trace(
            go.Scatter(
                x=[0, r_line * np.sin(az_rad)],
                y=[0, r_line * np.cos(az_rad)],
                mode="lines",
                line=dict(color="black", dash="dash", width=2),
                name=f"azimuth cau hinh ({az_val:.0f} do)",
            ),
            row=1,
            col=3,
        )

    fig.update_xaxes(title_text="Huong (do)", row=1, col=1)
    fig.update_yaxes(title_text="Vong ban kinh", row=1, col=1)
    fig.update_xaxes(title_text="Dong Tay (m)", row=1, col=3)
    fig.update_yaxes(title_text="Bac Nam (m)", row=1, col=3, scaleanchor="x3", scaleratio=1)

    # Zoom the rose to where the data actually is: the ring containing the
    # 97th percentile of samples, plus one ring of padding - a stray 1-2
    # sample outlier ring far out shouldn't compress everything else into
    # the plot's center.
    count_mat = _ring_direction_matrix(rd_row, cfg, "count")
    ring_totals = np.nan_to_num(np.nansum(count_mat, axis=1))
    display_edges = _ring_display_edges(cfg)
    total = ring_totals.sum()
    if total > 0:
        cum_frac = np.cumsum(ring_totals) / total
        cutoff_ring = int(np.searchsorted(cum_frac, 0.97))
    else:
        cutoff_ring = 0
    max_r = display_edges[min(cutoff_ring + 2, len(display_edges) - 1)]

    fig.update_layout(
        title=f"Cell {cell_id}: Coverage Fingerprint tu du lieu Mentor thuc te",
        template="plotly_white",
        height=560,
        width=1650,
        coloraxis=dict(colorscale="RdYlGn", cmin=cmin, cmax=cmax, colorbar=dict(title="RSRP dBm", x=1.06)),
        polar=dict(
            radialaxis=dict(title="m", range=[0, max_r]),
            angularaxis=dict(direction="clockwise", rotation=90, tickmode="array", tickvals=list(range(0, 360, 30))),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.86),
        margin=dict(t=90, b=100),
    )
    return fig
