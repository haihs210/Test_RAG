import pandas as pd

from coverage_intelligence.loader_cell_config import attach_bearing, attach_real_bearing, load_cell_config_xlsx


def _write_cell_config_xlsx(path, rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "Tên trên hệ thống", "Longtitude", "Latitude", "pci", "tac", "lcrid", "Băng tần",
            "Antenna gain", "Antenna high", "Mechaincal tilt", "Electrical tilt", "Total tilt",
            "azimuth", "Trạng thái hoạt động",
        ],
    )
    df.to_excel(path, index=False)


def test_load_cell_config_xlsx_projects_coordinates(tmp_path):
    path = tmp_path / "cells.xlsx"
    _write_cell_config_xlsx(
        path,
        [["4G-Q01136M12-HCM", 106.69014, 10.77215, None, 6132, None, "L1800", 18, 24.0, 0.0, 6.0, 6.0, 90.0, "Onair"]],
    )
    cfg = load_cell_config_xlsx(str(path))
    assert len(cfg) == 1
    row = cfg.iloc[0]
    assert row["cell_id"] == "4G-Q01136M12-HCM"
    assert row["site_id"] == "4G-Q01136-HCM"
    # known-good UTM48N projection for this lat/lon, from an independent check
    assert abs(row["site_x_m"] - 684802.8) < 5
    assert abs(row["site_y_m"] - 1191295.7) < 5


def test_attach_real_bearing_matches_known_geometry(tmp_path):
    path = tmp_path / "cells.xlsx"
    _write_cell_config_xlsx(
        path,
        [["CELL_A", 106.69014, 10.77215, None, 1, None, "L1800", 18, 24.0, 0.0, 6.0, 6.0, 0.0, "Onair"]],
    )
    cfg = load_cell_config_xlsx(str(path))
    site_x, site_y = cfg.iloc[0]["site_x_m"], cfg.iloc[0]["site_y_m"]

    # a UE due east of the site should get bearing ~90 degrees
    ue = pd.DataFrame({"cell_id": ["CELL_A"], "ue_x_m": [site_x + 100.0], "ue_y_m": [site_y]})
    out = attach_real_bearing(ue, cfg)
    assert abs(out.iloc[0]["bearing_deg"] - 90.0) < 1e-6


def test_attach_bearing_falls_back_for_unmatched_cells(tmp_path):
    path = tmp_path / "cells.xlsx"
    _write_cell_config_xlsx(
        path,
        [["CELL_A", 106.69014, 10.77215, None, 1, None, "L1800", 18, 24.0, 0.0, 6.0, 6.0, 0.0, "Onair"]],
    )
    cfg = load_cell_config_xlsx(str(path))

    ue = pd.DataFrame(
        {
            "cell_id": ["CELL_B", "CELL_B"],
            "ue_x_m": [1000.0, 1000.0],
            "ue_y_m": [2000.0, 2000.0],
            "distance_m": [5.0, 500.0],
        }
    )
    out = attach_bearing(ue, cfg, fallback_max_distance_m=60.0)
    # CELL_B is absent from the config; the close-in sample (5m) anchors the
    # site-position estimate, so both rows should still get a bearing.
    assert out["bearing_deg"].notna().all()
