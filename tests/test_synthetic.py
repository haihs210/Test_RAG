import numpy as np

from coverage_intelligence.synthetic import generate_demo_dataset


def test_generate_demo_dataset_shapes():
    cells, scenarios, ue, alarms = generate_demo_dataset(n_cells=20, n_days=5, base_samples_per_cell_day=100, seed=7)

    assert len(cells) == 20
    assert set(["cell_id", "site_id", "band", "tech", "azimuth_deg", "x_m", "y_m", "peer_group", "province"]).issubset(cells.columns)
    assert cells["cell_id"].is_unique

    assert len(ue) > 0
    assert set(["cell_id", "site_id", "date", "distance_m", "bearing_deg", "rsrp_dbm"]).issubset(ue.columns)
    assert ue["rsrp_dbm"].between(-140, -44).all()
    assert (ue["distance_m"] >= 0).all()
    assert ue["bearing_deg"].between(0, 360).all()

    assert len(scenarios) > 0
    assert set(alarms.columns) == {"alarm_id", "cell_id", "alarm_type", "severity", "raised_at", "cleared_at"}


def test_outage_scenario_collapses_sample_volume():
    cells, scenarios, ue, alarms = generate_demo_dataset(n_cells=30, n_days=10, base_samples_per_cell_day=200, seed=1)
    outage = next((s for s in scenarios if s.kind == "OUTAGE"), None)
    assert outage is not None

    before = ue[(ue.cell_id == outage.cell_id) & (ue.date < ue.date.min() + np.timedelta64(outage.start_day, "D"))]
    after = ue[(ue.cell_id == outage.cell_id) & (ue.date >= ue.date.min() + np.timedelta64(outage.start_day, "D"))]
    days_before = max((before.date.max() - before.date.min()).days + 1, 1) if len(before) else 1
    days_after = max((after.date.max() - after.date.min()).days + 1, 1) if len(after) else 1
    assert (len(after) / days_after) < (len(before) / days_before) * 0.5
