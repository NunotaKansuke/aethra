# aethra

**a**utomated **e**vent **t**racker and **c**haracterizer for **r**oman **a**lerts

A configurable pipeline for detecting **microlensing events** in photometric
light curves. It works with any dataset that has time, magnitude, and
magnitude-error columns (column names are configurable), automatically handling
file-format detection, multi-object grouping, observing-season splitting, and
multi-band ("achromatic") vetoes.

## What it does

Detection is model-free and classification comes after, which is the pipeline's
main design decision. A planetary or binary anomaly is by definition a
departure from a point-lens model, so deciding what is a candidate by how well
that model fits throws away the events most worth finding.

For each object the pipeline:

1. **Detects** a coherent brightening over the whole light curve — daily
   binning, robust normalization, multi-scale smoothing — without fitting any
   model and without scoping to one observing season (`peak_score`).
2. **Renormalizes the errors** from the scatter of the baseline outside the
   event.
3. **Fits a PSPL model** to the whole curve, started from an FFT matched
   filter that searches t0 globally.
4. **Scans the fit residuals** with the same coherence machinery: structure
   localized at t0 means an anomaly, diffuse structure means systematics or a
   variable.
5. **Labels** the object — see `label` under Output columns. Variable stars are
   identified by confident periodicity or by a chromatic brightening; a bad
   PSPL fit routes an object to `non_pspl_*`, it never discards it.

### Measured

On 2371 simulated Roman events, 11376 `roman_variable` light curves, and 2369
real Roman light curves with the event seasons removed — the same objects
through both pipelines, F146 only:

| | recall (events) | false positives (variables) | false positives (quiet stars) |
|---|---|---|---|
| season-scoped χ² gate | 0.822 | 0.0768 (874/11376) | — |
| this pipeline | 0.934 | 0.0003 (3/11376) | 0.0000 (0/2369) |

The gate found 152 events this pipeline misses, at a median true peak
amplitude of 0.09 mag; this pipeline finds 418 the gate missed, at a median of
2.45 mag. Recall holds above 0.98 while the event occupies less than a tenth
of the observed baseline and falls off past a fifth of it — a timescale long
enough to leave no baseline to measure against is the honest limit of a
model-free detector. Details and the reproduction scripts are in `.note/`.

## Installation
for the most updated version:

```bash
git clone https://github.com/rges-pit/aethra.git
cd aethra
pip install -e .
```
(Note: don't forget the . after the -e)

or stable version (note: this is not working yet): 

```
pip install aethra
```
Reading Parquet input needs the optional extra:

```bash
pip install -e ".[parquet]"
```

For development (tests + linting):

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.9. Core dependencies: NumPy, pandas, SciPy, Astropy.

## Quick start

```python
from aethra import load_and_run

config = {
    "time_col": "bjd",
    "mag_col":  "mag",
    "err_col":  "mag_err",
    "group_col": "name",   # column identifying each object in the table
}

results = load_and_run("data.parquet", config)

candidates = results[results["event_candidate"]]          # everything event-like
anomalies  = results[results["label"] == "non_pspl_candidate"]  # planet/binary
```

`load_and_run` auto-dispatches on the input type:

| `input_path`                                | Interpreted as |
|---------------------------------------------|----------------|
| `pd.DataFrame`                              | One table; objects split by `group_col` |
| `"data.parquet"` / `.fits` / `.csv` / `.txt`| One table file |
| `"lc/*.txt"` (glob)                         | One light curve per matched file |
| `("lc/*_W149.txt", "lc/*_Z087.txt")`        | Paired per-filter files matched by filename stem |

You can also call the per-DataFrame driver directly:

```python
from aethra import run_pipeline_from_dataframe
results = run_pipeline_from_dataframe(df, config)
```

### RGES `roman_variable` FITS files

The RGES variable-star dataset stores one filter per FITS extension and keeps
the time arrays in its `top_level` directory. Use the dedicated loader for one
FITS light curve:

```python
from aethra import load_roman_variable, run_pipeline_from_dataframe

lc = load_roman_variable(
    "../roman_variable/RGES_filters_CEP_lightcurves/"
    "RGES_filters_OGLE-BLG-CEP-019_lightcurves.fits"
)
config = {
    "time_col": "bjd",
    "mag_col": "mag",
    "err_col": "mag_err",
    "group_col": "name",
    "filter_col": "filt",
    "target_filter": "F146",
    "primary_filter": "F146",
    "secondary_filters": ["F087", "F213"],
}
results = run_pipeline_from_dataframe(lc, config)
```

The loader automatically finds `roman_variable/top_level` above the FITS file;
pass `time_dir=...` when the FITS file and time arrays are stored separately.

### Plotting a light curve and its PSPL fit

Install the optional plotting dependency:

```bash
pip install -e ".[plotting]"
```

Then pass the data for the event (or for one observing season) to the regular
Matplotlib helper:

```python
from aethra import load_roman_variable, plot_pspl_fit

lc = load_roman_variable("event.fits")
primary = lc[lc["filt"] == "F146"]
fig, axes, fit = plot_pspl_fit(
    primary["bjd"], primary["mag"], primary["mag_err"],
    title="OGLE-BLG-CEP-019 — F146",
    save_path="pspl-fit.png",
)
```

The top axis shows the observed magnitudes and PSPL curve; the bottom axis
shows `observed - model` residuals.

To compare representative variability classes in `roman_variable`, run:

```bash
python examples/plot_roman_variable_gallery.py
```

## Configuration from a YAML file

Rather than writing the `config` dict inline, you can keep all settings in a
YAML file and version it alongside your results — so every run records exactly
how it was configured. See [`examples/config.yaml`](examples/config.yaml) for a
fully commented template.

```python
from aethra import load_config, load_and_run

config = load_config("config.yaml")
results = load_and_run("data.parquet", config)
```

Keys you omit fall back to the built-in defaults; YAML `null` maps to Python
`None` (e.g. `group_col: null` means one object per file).


## Configuration reference

Passed as the `config` dict (or the matching CLI flag).

### Required

| Key        | Meaning                          |
|------------|----------------------------------|
| `time_col` | Time column (e.g. BJD)           |
| `mag_col`  | Magnitude column                 |
| `err_col`  | Magnitude-uncertainty column     |

### Grouping & input parsing

| Key        | Default | Meaning |
|------------|---------|---------|
| `group_col`| `None`  | Column whose unique values identify each source. `None` = one object per file/DataFrame. |
| `sep`      | `r"\s+"`| Separator regex for text files (`","` for CSV). |
| `header`   | `None`  | Header row index for text files; `None` = no header. |
| `columns`  | `None`  | Column names to assign when there is no header row. |

### Filters / achromatic test

| Key                | Default        | Meaning |
|--------------------|----------------|---------|
| `filter_col`       | `None`         | Band column. `None` skips the achromatic test. |
| `target_filter`    | `"F146"`       | Band used for event detection. |
| `primary_filter`   | `"F146"`       | Primary band in the achromatic test. |
| `secondary_filters`| `None`         | One band (str) or several (list) to compare against. |

### Tuning

| Key                   | Default | Meaning |
|-----------------------|---------|---------|
| `min_points`             | `10`   | Minimum points in the primary band to analyze. |
| `season_gap_days`        | `100`  | Day gap that separates observing seasons (reporting only). |
| `min_peak_score`         | `40.0` | Detection threshold on `peak_score`. The one knob that decides what is found. |
| `max_error_renorm`       | `1000.0` | Above this the baseline is not a baseline and the object is a variable star. |
| `residual_min_peak_score`| `40.0` | How strong residual structure must be to count as real. |
| `residual_localization_tE`| `2.0` | Residual structure within this many tE of t0 counts as localized. |
| `max_tE_over_duration`   | `3.0` | A fitted tE this many times the model-free excursion width is flagged degenerate. |
| `ffp_tE_max`             | `2.0`  | Max tE (days) to flag a free-floating-planet candidate. |
| `chromatic_min_points`   | `5`    | Min points per band for the achromatic test. |

`good_pspl_chi2` is no longer read: PSPL quality is reported, not gated on.

## Output columns

`label` carries the verdict. It is one of:

| Label | Meaning |
|---|---|
| `no_event` | Nothing coherent rose above the noise |
| `variable_star` | Confidently periodic, or the brightening is chromatic |
| `fit_failed` | An event, but no PSPL fit converged at all |
| `pspl_like` | An event PSPL describes, with structureless residuals |
| `non_pspl_unexplained` | An event whose residual structure is diffuse — inspect |
| `non_pspl_candidate` | An event whose residual structure is localized at t0 — anomaly |

The last three are the events. `event_candidate` is their union and
`is_candidate` is an alias of it, kept for callers written against the old
schema — but note it no longer means "PSPL fit well".

| Column | Type | Description |
|---|---|---|
| `name` | str | Object identifier |
| `label` | str | The verdict, as above |
| `is_candidate` / `event_candidate` | bool | An event happened |
| `is_ffp_candidate` | bool | Candidate with tE < `ffp_tE_max` days |
| `is_variable_star` | bool | Periodic or chromatic |
| `pspl_like`, `non_pspl_candidate`, `non_pspl_unexplained`, `fit_failed` | bool | `label` as indicators |
| `scan_candidate` | bool | Detection fired (`peak_score > min_peak_score`) |
| `peak_score` | float | Multi-scale coherence score — the detection statistic |
| `peak_sigma` | float | Peak height in robust sigma |
| `peak_time`, `onset_time` | float | Detected peak and start of the excursion (raw time units) |
| `n_up` | int | Coherent positive excursions found |
| `pos_ratio`, `duty_cycle` | float | Share of excess that is positive; share of nights in excursions |
| `main_duration_days`, `duration_fraction` | float | Extent of the main excursion, absolute and relative to the baseline |
| `n_nights` | int | Nights with data |
| `hc_periodic` | bool | Confidently periodic on the event-masked curve |
| `period` | float | Best period found (days) |
| `veto_periodic` | bool | Same as `hc_periodic` |
| `veto_recurrent` | bool | `n_up > 1`. **Reported only** — it does not veto anything |
| `veto_chromatic` | bool | The brightening disagrees across bands |
| `veto_baseline_variable` | bool | `error_renorm > max_error_renorm` — the curve varies away from the event too |
| `is_achromatic` | bool/nan | Achromatic test (`nan` = inconclusive) |
| `error_renorm` | float | Error scale factor from the baseline (floored at 1) |
| `baseline_chi2_red` | float | Reduced χ² of a constant fit to the baseline |
| `t0_fit` | float | PSPL best-fit peak time (HJD − 2450000) |
| `u0_fit` | float | PSPL best-fit impact parameter |
| `tE_fit` | float | PSPL best-fit Einstein crossing time (days) |
| `chi2_red_pspl` | float | Reduced χ² of the PSPL fit |
| `chi2_red_pspl_renorm` | float | The same divided by `error_renorm`. **Not a filter** — on a variable star the "baseline" is not baseline and this reaches values that pass any threshold |
| `frac_explained` | float | `1 − χ²_pspl/χ²_flat` |
| `fit_degenerate` | bool | `tE_fit > max_tE_over_duration × main_duration_days` — the fit slid down the tE↔u0 valley and the timescale is not determined by the data. **Reported only**, the event is still an event |
| `residual_peak_score` | float | Coherence score of the fit residuals |
| `residual_offset_tE` | float | Distance from t0 to the residual peak, in tE |
| `residual_significant` | bool | The residual structure is real |
| `residual_localized` | bool | ...and sits within `residual_localization_tE` of t0 |
| `n_seasons` | int | Observing seasons found |
| `best_season` | int | Season containing the detected peak |
| `is_flat` | bool | Light curve is consistent with a flat baseline |
| `chi2_flat`, `dof_flat`, `chi2_red_flat` | float | Constant-model fit over the whole curve |
| `bump_flag` | bool | Same as `scan_candidate` |
| `bump_snr` | float | Same as `peak_sigma` |
| `baseline_mag` | float | Median magnitude outside the event |
| `peak_mag` | float | Brightest magnitude within it |

## Package layout

```
src/aethra/
├── __init__.py      # public API
├── schema.py        # OUTPUT_COLUMNS, LABELS
├── config.py        # load_config (YAML → config dict)
├── coherence.py     # model-free detection: peak_score, excursions (Stage 2)
├── variability.py   # periodicity, whole-curve and event-masked
├── achromatic.py    # multi-band achromaticity test
├── pspl.py          # PSPL model, whole-curve fit, error renorm, residuals
├── pspl_seed.py     # FFT matched filter for (t0, tE, u0) starting values
├── detection.py     # legacy per-season bump detection and recurrence veto
├── seasons.py       # season splitting and per-season scan
├── plotting.py      # plot_pspl_fit
├── roman_variable.py# RGES variable-star FITS loader
├── pipeline.py      # run_pipeline_from_dataframe (main driver)
├── io.py            # file loaders + load_and_run dispatcher
└── cli.py           # `aethra` console script
```


## License

MIT — see [LICENSE](LICENSE).
