import numpy as np

from coverage_intelligence.config import CFSCDConfig
from coverage_intelligence.features import compute_fingerprints
from coverage_intelligence.synthetic import generate_demo_dataset


def test_fingerprint_densities_sum_to_one():
    cfg = CFSCDConfig()
    _, _, ue, _ = generate_demo_dataset(n_cells=15, n_days=4, base_samples_per_cell_day=300, seed=2)
    fp = compute_fingerprints(ue, cfg)

    ring_dens_cols = [c for c in fp.ring.columns if c.endswith("_density")]
    totals = fp.ring[ring_dens_cols].sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-6)

    dir_dens_cols = [c for c in fp.direction.columns if c.endswith("_density")]
    totals_dir = fp.direction[dir_dens_cols].sum(axis=1)
    assert np.allclose(totals_dir, 1.0, atol=1e-6)


def test_scalar_features_present_and_sane():
    cfg = CFSCDConfig()
    _, _, ue, _ = generate_demo_dataset(n_cells=10, n_days=3, base_samples_per_cell_day=200, seed=3)
    fp = compute_fingerprints(ue, cfg)

    for col in ["sample_count", "rsrp_mean", "rsrp_median", "dist_mean", "dist_p90"]:
        assert col in fp.scalar.columns
    assert (fp.scalar["sample_count"] > 0).all()
    assert fp.scalar["rsrp_mean"].between(-140, -44).all()
    assert (fp.scalar["rsrp_p90"] >= fp.scalar["rsrp_p10"]).all()
