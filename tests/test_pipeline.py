"""End-to-end check: does the pipeline detect (and mostly correctly explain)
each of the five anomaly scenarios from the proposal, without flooding
unrelated cells with false alarms once the baseline has warmed up?
"""

import warnings

import pandas as pd
import pytest

from coverage_intelligence import CFSCDConfig, CoverageIntelligencePipeline
from coverage_intelligence.synthetic import generate_demo_dataset

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def pipeline_result():
    cells, scenarios, ue, alarms = generate_demo_dataset(
        n_cells=80, n_days=35, base_samples_per_cell_day=1500, seed=3
    )
    pipeline = CoverageIntelligencePipeline(CFSCDConfig())
    result = pipeline.run(ue, cells, alarms)
    return cells, scenarios, ue, alarms, result


def _alerts_with_scenario(result, scenarios):
    scen_map = {s.cell_id: s for s in scenarios}
    alerts = result.alerts.copy()
    min_date = alerts["date"].min() if not alerts.empty else pd.Timestamp("2026-06-01")
    alerts["day_idx"] = (alerts["date"] - min_date).dt.days
    alerts["scenario_kind"] = alerts["cell_id"].map(lambda c: scen_map[c].kind if c in scen_map else "NONE")
    alerts["scenario_start"] = alerts["cell_id"].map(lambda c: scen_map[c].start_day if c in scen_map else -1)
    return alerts


@pytest.mark.parametrize("kind", ["OUTAGE", "SHADOWING", "TILT_POWER_DRIFT"])
def test_scenario_is_detected_during_incident_window(pipeline_result, kind):
    cells, scenarios, ue, alarms, result = pipeline_result
    alerts = _alerts_with_scenario(result, scenarios)
    scen_cells = [s.cell_id for s in scenarios if s.kind == kind]
    assert scen_cells, f"no synthetic {kind} scenario was generated"

    for cell_id in scen_cells:
        spec = next(s for s in scenarios if s.cell_id == cell_id)
        sub = alerts[(alerts.cell_id == cell_id) & (alerts.day_idx >= spec.start_day)]
        assert len(sub) >= 3, f"{kind} on {cell_id} was not detected (only {len(sub)} alerts after day {spec.start_day})"


@pytest.mark.parametrize("kind", ["OUTAGE", "SHADOWING", "TILT_POWER_DRIFT"])
def test_scenario_root_cause_majority_correct(pipeline_result, kind):
    cells, scenarios, ue, alarms, result = pipeline_result
    alerts = _alerts_with_scenario(result, scenarios)
    sub = alerts[(alerts.scenario_kind == kind) & (alerts.day_idx >= alerts["scenario_start"])]
    assert len(sub) > 0
    top = sub["top_cause"].value_counts().idxmax()
    assert top == kind, f"expected majority RCA cause {kind}, got distribution {sub['top_cause'].value_counts().to_dict()}"


def test_false_positive_rate_after_baseline_warmup(pipeline_result):
    cells, scenarios, ue, alarms, result = pipeline_result
    alerts = _alerts_with_scenario(result, scenarios)
    warmed_up = alerts[(alerts.scenario_kind == "NONE") & (alerts.day_idx >= 15)]
    total_normal_cell_days = cells["cell_id"].nunique() * 20  # days 15..34
    assert len(warmed_up) / total_normal_cell_days < 0.02


def test_alerts_have_actionable_fields(pipeline_result):
    cells, scenarios, ue, alarms, result = pipeline_result
    assert not result.alerts.empty
    for col in ["chs", "chs_class", "top_cause", "recommendation", "spatial_scope"]:
        assert col in result.alerts.columns
    assert result.alerts["chs"].between(0, 100).all()
