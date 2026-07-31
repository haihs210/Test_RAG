from coverage_intelligence.loader_mentor import _derive_site_id, load_mentor_export


def test_derive_site_id_named_cell():
    assert _derive_site_id("4G-Q01136M12-HCM") == "4G-Q01136-HCM"


def test_derive_site_id_cgi_fallback():
    assert _derive_site_id("452-2-6104991-11") == "452-2-6104991"


def test_derive_site_id_unparseable_falls_back_to_input():
    assert _derive_site_id("garbage") == "garbage"


def test_load_mentor_export_on_sample(tmp_path):
    raw = "\n".join(
        [
            "Call Index\tCall Start Time\tCall End Time\tMobile ID\tSubscriber ID\tRecord Type\tTimestamp\tTechnology\tConnection Status\tX\tY\tSector Carrier_0\tEC_0\tEC\\IO_0\tACTIVE_0\tCalibrated Distance_0",
            "1\t0\t0\t0\t0\tPOWER\t1000\tLTE\tCONNECTED\t100.0\t200.0\t4G-Q01136M12-HCM\t-95.0\t-10.0\tACTIVE\tN/A",
            "1\t0\t0\t0\t0\tDISTANCE\t1005\tLTE\tCONNECTED\t100.0\t200.0\t4G-Q01136M12-HCM\tN/A\tN/A\tACTIVE\t123.4",
        ]
    )
    path = tmp_path / "sample.txt"
    path.write_text(raw)

    ue = load_mentor_export(str(path))
    assert len(ue) == 1
    row = ue.iloc[0]
    assert row["cell_id"] == "4G-Q01136M12-HCM"
    assert row["site_id"] == "4G-Q01136-HCM"
    assert row["rsrp_dbm"] == -95.0
    assert row["distance_m"] == 123.4
