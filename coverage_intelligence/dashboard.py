"""Builds the self-contained HTML "Visualization & Operation Layer" dashboard.

Mirrors what the proposal describes: Coverage Health Score by hierarchy
level, alert list with recommended actions, and a Coverage Fingerprint
map (Ring x Direction) showing an anomalous cell before/after the change -
e.g. the antenna-misalignment example in Hinh 4 of the proposal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CFSCDConfig
from .pipeline import PipelineResult

CLASS_COLORS = {
    "Excellent": "#2e7d32",
    "Good": "#8bc34a",
    "Warning": "#f9a825",
    "Critical": "#c62828",
}


def _kpi_cards_html(res: PipelineResult) -> str:
    chs = res.chs.join(res.fingerprints.scalar[["sample_count"]])
    latest_date = res.fingerprints.scalar.index.get_level_values("date").max()
    latest = chs.xs(latest_date, level="date")
    counts = latest["chs_class"].value_counts()
    n_cells = res.cells["cell_id"].nunique()
    n_alerts = 0 if res.alerts.empty else len(res.alerts)
    network_chs = (latest["chs"] * latest["sample_count"]).sum() / max(latest["sample_count"].sum(), 1)

    cards = [
        ("Toan mang CHS", f"{network_chs:.1f}", "#37474f"),
        ("So cell", f"{n_cells}", "#37474f"),
        ("Canh bao dang hoat dong", f"{n_alerts}", "#37474f"),
    ]
    for cls in ["Excellent", "Good", "Warning", "Critical"]:
        cards.append((cls, str(int(counts.get(cls, 0))), CLASS_COLORS[cls]))

    cells_html = "".join(
        f'<div class="kpi-card"><div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>'
        for label, value, color in cards
    )
    return f'<div class="kpi-row">{cells_html}</div>'


def _fig_chs_trend(res: PipelineResult, level_col: str) -> go.Figure:
    agg = res.aggregate(level_col).reset_index()
    fig = go.Figure()
    for level, sub in agg.groupby(level_col):
        sub = sub.sort_values("date")
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["chs"], mode="lines+markers", name=str(level)))
    fig.update_layout(
        title=f"Xu huong Coverage Health Score theo {level_col}",
        xaxis_title="Ngay",
        yaxis_title="CHS (0-100)",
        template="plotly_white",
        height=420,
    )
    fig.add_hrect(y0=0, y1=50, fillcolor=CLASS_COLORS["Critical"], opacity=0.06, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor=CLASS_COLORS["Warning"], opacity=0.06, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor=CLASS_COLORS["Good"], opacity=0.06, line_width=0)
    return fig


def _fig_chs_histogram(res: PipelineResult) -> go.Figure:
    latest_date = res.fingerprints.scalar.index.get_level_values("date").max()
    latest = res.chs.xs(latest_date, level="date")
    fig = go.Figure()
    for cls, color in CLASS_COLORS.items():
        vals = latest.loc[latest["chs_class"] == cls, "chs"]
        fig.add_trace(go.Histogram(x=vals, name=cls, marker_color=color, opacity=0.85))
    fig.update_layout(
        barmode="stack",
        title=f"Phan bo Coverage Health Score toan mang ({latest_date.date()})",
        xaxis_title="CHS",
        yaxis_title="So cell",
        template="plotly_white",
        height=380,
    )
    return fig


def _fig_cell_timeline(res: PipelineResult, cfg: CFSCDConfig, cell_id: str) -> Optional[go.Figure]:
    scalar = res.fingerprints.scalar
    if cell_id not in scalar.index.get_level_values("cell_id"):
        return None
    sub = scalar.loc[cell_id].sort_index()
    chs_sub = res.chs.loc[cell_id].sort_index() if cell_id in res.chs.index.get_level_values(0) else None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=sub.index, y=sub["rsrp_mean"], mode="lines+markers", name="RSRP trung binh thuc te", line=dict(color="#1565c0")),
        secondary_y=False,
    )
    if chs_sub is not None:
        fig.add_trace(
            go.Scatter(x=chs_sub.index, y=chs_sub["chs"], mode="lines", name="Coverage Health Score", line=dict(color="#c62828", dash="dot")),
            secondary_y=True,
        )
    fig.update_layout(
        title=f"Cell {cell_id}: RSRP trung binh va Coverage Health Score theo ngay",
        template="plotly_white",
        height=380,
    )
    fig.update_yaxes(title_text="RSRP trung binh (dBm)", secondary_y=False)
    fig.update_yaxes(title_text="CHS (0-100)", range=[0, 100], secondary_y=True)
    return fig


def _ring_direction_matrix(row: pd.Series, cfg: CFSCDConfig, metric: str = "mean") -> np.ndarray:
    n_rings, n_dirs = len(cfg.ring_edges_m), cfg.n_direction_sectors
    mat = np.full((n_rings, n_dirs), np.nan)
    for r in range(n_rings):
        for d in range(n_dirs):
            col = f"rd_{r * n_dirs + d}_{metric}"
            if col in row.index:
                mat[r, d] = row[col]
    return mat


def _fig_fingerprint_heatmap(res: PipelineResult, cfg: CFSCDConfig, cell_id: str, alert_date, cause: str) -> Optional[go.Figure]:
    rd = res.fingerprints.ring_direction
    if cell_id not in rd.index.get_level_values("cell_id"):
        return None
    sub = rd.loc[cell_id].sort_index()
    dates = sub.index
    before_candidates = dates[dates < alert_date]
    if len(before_candidates) == 0:
        return None
    before_date = before_candidates[max(0, len(before_candidates) - 8)]

    before_mat = _ring_direction_matrix(sub.loc[before_date], cfg)
    after_mat = _ring_direction_matrix(sub.loc[alert_date], cfg)

    ring_labels = [f"{int(cfg.ring_edges_m[i])}-{'inf' if i + 1 == len(cfg.ring_edges_m) else int(cfg.ring_edges_m[i+1])}m" for i in range(len(cfg.ring_edges_m))]
    dir_labels = [f"{int(i * 360 / cfg.n_direction_sectors)}" for i in range(cfg.n_direction_sectors)]

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=(f"Truoc ({before_date.date()})", f"Sau - nghi ngo {cause} ({alert_date.date()})")
    )
    zmin = np.nanmin([before_mat, after_mat])
    zmax = np.nanmax([before_mat, after_mat])
    fig.add_trace(
        go.Heatmap(z=before_mat, x=dir_labels, y=ring_labels, colorscale="RdYlGn", zmin=zmin, zmax=zmax, coloraxis="coloraxis"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=after_mat, x=dir_labels, y=ring_labels, colorscale="RdYlGn", zmin=zmin, zmax=zmax, coloraxis="coloraxis"),
        row=1, col=2,
    )
    fig.update_layout(
        title=f"Cell {cell_id}: Coverage Fingerprint (Ring x Direction, RSRP trung binh dBm)",
        coloraxis=dict(colorscale="RdYlGn", cmin=zmin, cmax=zmax, colorbar=dict(title="dBm")),
        template="plotly_white",
        height=420,
    )
    fig.update_xaxes(title_text="Huong (do)")
    fig.update_yaxes(title_text="Vong ban kinh")
    return fig


def _alerts_table_html(res: PipelineResult, top_n: int = 30) -> str:
    if res.alerts.empty:
        return "<p>Khong co canh bao nao.</p>"
    cols = [
        "date", "cell_id", "site_id", "province", "band", "chs", "chs_class",
        "top_cause", "top_confidence", "spatial_scope", "recommendation",
    ]
    cols = [c for c in cols if c in res.alerts.columns]
    view = res.alerts.sort_values("chs").head(top_n)[cols].copy()
    if "date" in view:
        view["date"] = pd.to_datetime(view["date"]).dt.strftime("%Y-%m-%d")
    view["chs"] = view["chs"].round(1)
    if "top_confidence" in view:
        view["top_confidence"] = view["top_confidence"].round(2)

    def row_html(r):
        cls = r.get("chs_class", "")
        color = CLASS_COLORS.get(cls, "#333")
        cells = "".join(f"<td>{r[c]}</td>" for c in cols)
        return f'<tr style="border-left:4px solid {color}">{cells}</tr>'

    header = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(row_html(r) for _, r in view.iterrows())
    return f'<table class="alerts-table"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'


_PAGE_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 0; background: #fafafa; color: #212121; }
header { background: #0d47a1; color: white; padding: 20px 32px; }
header h1 { margin: 0 0 4px 0; font-size: 22px; }
header p { margin: 0; opacity: 0.85; font-size: 13px; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 32px 60px; }
section { background: white; border-radius: 10px; padding: 20px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
h2 { font-size: 16px; color: #37474f; margin-top: 0; }
.kpi-row { display: flex; gap: 16px; flex-wrap: wrap; }
.kpi-card { flex: 1; min-width: 110px; text-align: center; padding: 14px 8px; background: #f5f5f5; border-radius: 8px; }
.kpi-value { font-size: 26px; font-weight: 700; }
.kpi-label { font-size: 12px; color: #616161; margin-top: 4px; }
table.alerts-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
table.alerts-table th { text-align: left; background: #f0f2f5; padding: 8px; position: sticky; top: 0; }
table.alerts-table td { padding: 6px 8px; border-bottom: 1px solid #eee; }
"""


def build_dashboard(res: PipelineResult, cfg: CFSCDConfig, output_path: str, level_col: str = "province", n_examples: int = 4):
    figs_html = []
    first = True
    for fig in [_fig_chs_trend(res, level_col), _fig_chs_histogram(res)]:
        figs_html.append(fig.to_html(full_html=False, include_plotlyjs="inline" if first else False))
        first = False

    example_html = ""
    if not res.alerts.empty:
        by_cell = res.alerts.sort_values("chs").drop_duplicates(subset=["cell_id"])
        one_per_cause = by_cell.sort_values("chs").drop_duplicates(subset=["top_cause"])
        remaining = n_examples - len(one_per_cause)
        extra = by_cell[~by_cell["cell_id"].isin(one_per_cause["cell_id"])].head(max(remaining, 0))
        examples = pd.concat([one_per_cause, extra]).sort_values("chs").head(n_examples)
        for _, row in examples.iterrows():
            tl = _fig_cell_timeline(res, cfg, row["cell_id"])
            hm = _fig_fingerprint_heatmap(res, cfg, row["cell_id"], row["date"], row.get("top_cause", "?"))
            parts = []
            if tl is not None:
                parts.append(tl.to_html(full_html=False, include_plotlyjs=False))
            if hm is not None:
                parts.append(hm.to_html(full_html=False, include_plotlyjs=False))
            if parts:
                example_html += f'<section><h2>Vi du: {row["cell_id"]} ({row.get("top_cause","?")}, CHS={row["chs"]:.1f})</h2>' + "".join(parts) + "</section>"

    html = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<title>Coverage Intelligence Platform - Dashboard</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
<header>
  <h1>Coverage Intelligence Platform</h1>
  <p>CF-SCD - Coverage Fingerprint / Spatial Change Detection - du lieu mo phong (demo)</p>
</header>
<main>
  <section><h2>Tong quan mang</h2>{_kpi_cards_html(res)}</section>
  <section><h2>Xu huong Coverage Health Score</h2>{figs_html[0]}</section>
  <section><h2>Phan bo CHS toan mang</h2>{figs_html[1]}</section>
  <section><h2>Danh sach canh bao (uu tien theo CHS thap nhat)</h2>{_alerts_table_html(res)}</section>
  {example_html}
</main>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
