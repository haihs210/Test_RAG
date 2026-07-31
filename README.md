# Coverage Intelligence Platform - CF-SCD reference implementation

A working, testable implementation of the **CF-SCD (Coverage Fingerprint -
Spatial Change Detection)** algorithm described in
`HKVTKTM_Hien_ke_VPS_final.docx` - a proposal for turning VNPT's UE Report
data (via Mentor) into automatic radio-coverage anomaly detection, root-cause
analysis and a Coverage Health Score.

See `docs/algorithm_analysis.md` (Vietnamese) for the full algorithm
analysis this implementation is based on.

## What's here

```
coverage_intelligence/
  config.py       tunable thresholds (baseline window, detection thresholds, CHS weights, ...)
  synthetic.py     large-scale synthetic UE Report / cell RF config / Alarm generator
  loader_mentor.py  parses VNPT's real Mentor POWER/DISTANCE export into the same UE Report shape
  loader_cell_config.py  parses a real cell/site RF+installation export (lat/lon, azimuth, tilt) for exact geometry
  real_data_viz.py  2-panel per-cell figure (binned heatmap + real sample points) for inspecting real-data extraction
  features.py      Step 1-2: Coverage Feature Extraction -> Coverage Fingerprint
  baseline.py       Step 3: rolling baseline (mean/median/MAD) + peer-group fallback
  detection.py       Step 4: rule-based + Spatial Similarity Index + EWMA/CUSUM/STL-lite, ensemble voting
  rca.py               Step 5: neighbor correlation + expert rule library + alarm correlation
  health_score.py        Step 6: Coverage Health Score + hierarchy aggregation
  pipeline.py               orchestration (Steps 1-6) + FeedbackStore (Step 7, human-in-the-loop)
  dashboard.py                self-contained HTML dashboard (Plotly)
scripts/run_demo.py    synthetic data end-to-end: generate, run the pipeline, write alerts.csv + dashboard.html
scripts/run_real_data_demo.py  real Mentor export -> Coverage Fingerprint (steps 1-2 only, see below)
tests/                  pytest: feature/baseline sanity checks + end-to-end scenario detection
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_demo.py --n-cells 120 --n-days 35 --samples-per-cell-day 1500 --out out/
python -m pytest tests/ -q
```

This generates a synthetic dataset (~6M UE Report rows for the sizes above),
runs the full CF-SCD pipeline, and writes `out/alerts.csv` and
`out/dashboard.html` (open the latter in a browser).

## The five scenarios from the proposal, and how the pipeline catches them

The synthetic generator (`synthetic.py`) injects the same anomaly types the
proposal's "kich ban ung dung tieu bieu" table describes, and the test suite
(`tests/test_pipeline.py`) asserts each one is both **detected** and
**correctly explained**:

| Scenario | Injected as | Caught mainly via |
|---|---|---|
| RRU/Cell Outage | sample volume collapses to ~3%, remaining RSRP very poor | Rule-based (sample-drop) + Ensemble |
| Shadowing (che chan) | one bearing sector loses signal, others untouched, no sector improves | Spatial Similarity Index (SSI) + RCA's "frac_degraded / no gain elsewhere" pattern |
| Tilt / power drift | radius shrinks + RSRP drops together | Rule-based (radius+RSRP) + RCA's signed radius delta |
| Azimuth drift | some sectors improve as others degrade (beam redirected) | SSI + RCA's "frac_improved" / peak-bearing shift |
| Overshoot (new cell) | radius grows well beyond peer-group norm, too little own history | Peer-group baseline fallback + RCA's radius growth signal |

On this synthetic dataset the pipeline detects >80% of injected incident-days
per scenario with 0 false positives on unaffected cells once the baseline has
warmed up (first `baseline_window_days`), and gets the root cause right on
~80%+ of the alerts it raises. These are demo-scale numbers, not a claim
about production accuracy - see "Where a real deployment differs" below.

## Where a real deployment differs from this demo (and about that "large sample data" question)

I don't have direct access to VNPT's Mentor database, RIMS or nFM, so there
is no way for me to pull real data on my own. Two ways to close that gap,
both now wired up:

1. **You provide a real (or real-shaped) sample.** This has already been
   validated on two real VNPT exports:
   - `coverage_intelligence/loader_mentor.py` parses the raw Mentor
     "power/distance" export (tab-separated event log with `POWER` records
     carrying real RSRP - `EC_0` - and UE position, and `DISTANCE` records
     carrying real distance-to-site) into the UE-Report shape
     `features.compute_fingerprints` expects.
   - `coverage_intelligence/loader_cell_config.py` parses a real cell/site
     RF+installation export (Latitude/Longitude, azimuth, mechanical/
     electrical tilt, antenna height/gain, band, operational status - the
     same attributes the proposal names for the Data Acquisition Layer) and
     joins it onto the UE Report table to compute **exact** bearing-from-
     site instead of an estimate.

   With `--cell-config`, the loader also stops throwing away ~70% of the
   real signal-level data the export actually carries: each `POWER` row
   reports up to 12 cells at once (the serving cell in `EC_0`, plus up to 11
   candidate/neighbor cells in `EC_1..EC_11`), and every one of those is a
   real "mức thu" (received signal level) sample of *that* cell at the UE's
   position - not just the serving cell's own reading.
   `loader_mentor.load_mentor_power_measurements` extracts all of them, and
   `loader_cell_config.attach_geometry` computes `distance_m`/`bearing_deg`
   for every row geometrically from real site coordinates (candidate cells
   have no directly-logged `DISTANCE` record to fall back on, so this only
   works for cells present in the config - rows for anything else are
   dropped rather than guessed).

   Run both together with:
   ```bash
   python scripts/run_real_data_demo.py path/to/raw_mentor.txt \
       --cell-config path/to/cell_config.xlsx --out out_real/
   ```
   On the two samples provided (raw Mentor export + a 1654-row cell config
   export, both covering one hour in Ho Chi Minh City), this recovers
   **98,466 power measurements across 1,171 distinct cells** (serving +
   candidate) from what a serving-cell-only reading would see as ~30k
   measurements across ~450 cells; 69,181 of those measurements matched a
   cell with known site coordinates (535 distinct cells), each with an
   exact, geometrically-computed distance and bearing. That's enough
   density to support much finer Ring resolution too - the real-data script
   uses ~50m rings (vs. the pipeline's tuned default in `config.py`, sized
   for the sample density a single day of serving-cell-only UE Report
   normally has) and still gets populated Ring x Direction bins for the
   busier cells. For each of the best-covered cells it writes a 2-panel
   HTML (`out_real/cell_fingerprints/<cell_id>.html`): the binned Ring x
   Direction heatmap side-by-side with a scatter of the actual measurement
   points (real coordinates, colored by real RSRP, with the site position
   and configured azimuth overlaid) - useful for sanity-checking the
   heatmap against what it was actually built from. Run without
   `--cell-config` to fall back to serving-cell-only + estimated site
   position instead.

   One caveat remains, independent of which loader is used: **a single
   export is one snapshot in time.** Baseline/Detection/RCA/CHS (steps 3-6)
   need 7-30+ days of history per the proposal - a one-hour export only
   exercises steps 1-2 (Coverage Fingerprint extraction). To run the full
   pipeline on real data, repeat the same export daily and concatenate the
   parsed UE Report tables before calling
   `CoverageIntelligencePipeline.run()`.

2. **Scale via the synthetic generator.** `synthetic.py` is fully vectorized
   (`np.repeat` expansion, no per-sample Python loop), so it already
   produces multi-million-row datasets in seconds - useful for load testing
   the pipeline (`--n-cells 500 --n-days 30 --samples-per-cell-day 3000` is
   ~45M rows) independent of whether real data is available.

For production, the proposal's own architecture (section 3.2.1 / 3.3.4) is
the intended integration path and matches how this code is structured to
receive data:

- **UE Report**: periodic batch extract from the Mentor database (the
  proposal explicitly keeps this - CF-SCD is a layer *on top of* Mentor, not
  a replacement). At VNPT's actual data volumes this lands on Hadoop/Spark;
  `features.py`'s groupby/pivot logic is expressed in pandas here for
  readability but maps directly onto Spark's DataFrame API (same groupby +
  pivot operations) if/when raw UE Report no longer fits in memory on one
  machine.
- **Cell RF/installation config** (Cell ID, Site ID, band, PCI, ARFCN,
  Azimuth, tilt, height, tx power, coordinates): normalized file import or
  API pull from the network resource-management system (e.g. RIMS).
- **Alarm**: pulled via API from the alarm-management system (e.g. nFM),
  and - as the proposal specifies - *only* for cells already flagged
  anomalous, not the whole network, which is why `rca.py` only queries
  alarms for alert rows rather than joining the full alarm feed up front.

One design point worth calling out: the raw UE Report volume (potentially
hundreds of millions of rows/day network-wide) is reduced by
`features.compute_fingerprints` to one Coverage Fingerprint row per
(cell, day) *before* baseline/detection/RCA run. So the parts of the
pipeline downstream of feature extraction operate on a table sized
`n_cells x n_days`, not on the raw sample volume - this is what makes daily,
network-wide detection tractable, and it's a property of the proposal's
design, not something added here.

## Tuning

All thresholds live in `coverage_intelligence/config.py`
(`CFSCDConfig`) - baseline window (7-30 days per the proposal), rule-based
thresholds, SSI/EWMA/CUSUM parameters, ensemble vote count, RCA neighbor
radius, and CHS weights/classification cutoffs. They were calibrated against
the synthetic scenarios here and are meant to be recalibrated against real
Ground Truth, exactly as step 7 (`pipeline.FeedbackStore`) is meant to feed
back into `confirmed_incident_dates` on the next run.
