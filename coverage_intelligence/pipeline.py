"""End-to-end orchestration of the CF-SCD seven-step process.

    thu thap -> chuan hoa -> Coverage Fingerprint -> phat hien bien dong ->
    phan tich nguyen nhan -> Coverage Health Score -> Dashboard/canh bao ->
    ky su xac nhan -> he thong hoc hoi va cap nhat tri thuc

Steps 1-6 run inside :meth:`CoverageIntelligencePipeline.run`. Step 7
(Human-in-the-loop feedback) is :class:`FeedbackStore`, which is
deliberately separate: it is meant to be called after engineers verify
alerts, and its output (confirmed incident dates) feeds back into the next
run's baseline construction via ``confirmed_incident_dates``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd

from .baseline import compute_baseline, find_outlier_dates, peer_group_baseline
from .config import CFSCDConfig
from .detection import compute_ssi, ensemble_vote, merge_baselines, rule_based_flags, time_series_flags
from .features import FingerprintTable, compute_fingerprints
from .health_score import aggregate_health_score, compute_health_score
from .rca import build_neighbor_index, diagnose_alerts


@dataclass
class PipelineResult:
    fingerprints: FingerprintTable
    detection: pd.DataFrame
    rca: pd.DataFrame
    chs: pd.DataFrame
    alerts: pd.DataFrame
    cells: pd.DataFrame

    def aggregate(self, level_col: str = "province") -> pd.DataFrame:
        return aggregate_health_score(self.chs, self.fingerprints.scalar, self.cells, level_col)


class CoverageIntelligencePipeline:
    def __init__(self, cfg: Optional[CFSCDConfig] = None):
        self.cfg = cfg or CFSCDConfig()

    def run(
        self,
        ue_reports: pd.DataFrame,
        cells: pd.DataFrame,
        alarms: Optional[pd.DataFrame] = None,
        confirmed_incident_dates: Optional[Dict[str, Set]] = None,
    ) -> PipelineResult:
        cfg = self.cfg
        alarms = alarms if alarms is not None else pd.DataFrame(
            columns=["alarm_id", "cell_id", "alarm_type", "severity", "raised_at", "cleared_at"]
        )

        # Steps 1-2: Feature Extraction + Coverage Fingerprint
        fp = compute_fingerprints(ue_reports, cfg)
        scalar = fp.scalar.sort_index()
        direction = fp.direction.sort_index()
        ring_direction = fp.ring_direction.sort_index()

        # Step 3: Baseline Construction (own history + peer-group fallback)
        excluded = find_outlier_dates(scalar, cfg, confirmed_incident_dates)
        own_scalar = compute_baseline(scalar, cfg, excluded)
        own_direction = compute_baseline(direction, cfg, excluded)
        own_rd = compute_baseline(ring_direction, cfg, excluded)

        peer_map = cells.set_index("cell_id")["peer_group"]
        peer_scalar = peer_group_baseline(scalar, peer_map, cfg)
        peer_direction = peer_group_baseline(direction, peer_map, cfg)
        peer_rd = peer_group_baseline(ring_direction, peer_map, cfg)

        eff_scalar = merge_baselines(own_scalar, peer_scalar)
        eff_direction = merge_baselines(own_direction, peer_direction)
        eff_rd = merge_baselines(own_rd, peer_rd)

        # Step 4: Spatial Change Detection (rule-based + SSI + time-series -> ensemble)
        rule_df = rule_based_flags(scalar, eff_scalar, cfg)
        ssi_df = compute_ssi(ring_direction, eff_rd, cfg)
        ts_df = time_series_flags(scalar, eff_scalar, cfg)
        detection = ensemble_vote(rule_df, ssi_df, ts_df, cfg)
        detection["baseline_source"] = eff_scalar.source.reindex(detection.index)

        # Step 5: Root Cause Analysis & Recommendation
        neighbors = build_neighbor_index(cells, cfg)
        anomaly_lookup = detection["is_anomaly"].to_dict()
        alert_keys: List[tuple] = detection.index[detection["is_anomaly"]].tolist()
        rca_df = diagnose_alerts(
            alert_keys, detection, scalar, direction, eff_direction, cells, alarms, neighbors, anomaly_lookup, cfg
        )

        # Step 6: Coverage Health Score
        chs_df = compute_health_score(scalar, detection, rca_df, cfg)

        alerts = self._build_alert_table(detection, rca_df, chs_df, scalar, cells)

        return PipelineResult(fingerprints=fp, detection=detection, rca=rca_df, chs=chs_df, alerts=alerts, cells=cells)

    @staticmethod
    def _build_alert_table(
        detection: pd.DataFrame,
        rca_df: pd.DataFrame,
        chs_df: pd.DataFrame,
        scalar: pd.DataFrame,
        cells: pd.DataFrame,
    ) -> pd.DataFrame:
        anomalies = detection[detection["is_anomaly"]].copy()
        if anomalies.empty:
            return pd.DataFrame()
        alerts = anomalies.join(chs_df).join(scalar[["rsrp_mean", "sample_count", "dist_p90"]])
        alerts = alerts.reset_index()
        if not rca_df.empty:
            alerts = alerts.merge(rca_df, on=["cell_id", "date"], how="left", suffixes=("", "_rca"))
        alerts = alerts.merge(
            cells[["cell_id", "site_id", "band", "tech", "vendor", "province"]], on="cell_id", how="left"
        )
        return alerts.sort_values("chs")


@dataclass
class FeedbackStore:
    """Step 7: Human-in-the-loop feedback log.

    Engineers record whether an alert was a real, actionable coverage issue.
    Confirmed true positives feed back into the next run's baseline (their
    dates are excluded from the historical window so the incident doesn't
    pollute future baselines); confirmed false positives are the training
    signal the proposal describes for eventually moving the rule library
    towards a supervised model.
    """

    records: List[dict] = field(default_factory=list)

    def record(self, cell_id: str, date, confirmed_cause: str, is_true_positive: bool, notes: str = ""):
        self.records.append(
            dict(
                cell_id=cell_id,
                date=pd.Timestamp(date),
                confirmed_cause=confirmed_cause,
                is_true_positive=is_true_positive,
                notes=notes,
            )
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def confirmed_incident_dates(self) -> Dict[str, Set]:
        """True-positive dates, grouped by cell -> excluded from future baselines."""
        df = self.to_frame()
        if df.empty:
            return {}
        tp = df[df["is_true_positive"]]
        return tp.groupby("cell_id")["date"].apply(set).to_dict()

    def false_positive_rate(self) -> float:
        df = self.to_frame()
        if df.empty:
            return float("nan")
        return float((~df["is_true_positive"]).mean())
