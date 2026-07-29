"""Basic smoke and functional tests for the aethra pipeline."""

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

import aethra as ef
from aethra import (
    OUTPUT_COLUMNS,
    detect_bump,
    fit_pspl,
    load_and_run,
    pspl_magnification,
    run_pipeline_from_dataframe,
    split_into_seasons,
)
from aethra.roman_variable import load_roman_variable


def make_lightcurve(with_event=True, seed=0):
    """A two-season light curve, optionally with a PSPL bump in season 1."""
    rng = np.random.default_rng(seed)
    # Two seasons separated by a >100 day gap.
    t1 = np.linspace(2459000, 2459120, 400)
    t2 = np.linspace(2459300, 2459420, 400)
    time = np.concatenate([t1, t2])

    baseline = 18.0
    mag = np.full_like(time, baseline)

    if with_event:
        t0, u0, tE = 2459060.0, 0.1, 8.0
        A = pspl_magnification(time, t0, u0, tE)
        # brighter (smaller mag) at higher magnification
        mag = baseline - 2.5 * np.log10(A)

    mag_err = np.full_like(time, 0.01)
    mag = mag + rng.normal(0, 0.01, size=mag.size)

    return pd.DataFrame({"bjd": time, "mag": mag, "mag_err": mag_err, "name": "obj1"})


CONFIG = {
    "time_col": "bjd",
    "mag_col": "mag",
    "err_col": "mag_err",
    "group_col": "name",
}


def test_version_and_exports():
    assert isinstance(ef.__version__, str)
    assert "load_and_run" in ef.__all__


def test_output_schema_is_stable():
    df = run_pipeline_from_dataframe(make_lightcurve(), CONFIG)
    for col in OUTPUT_COLUMNS:
        assert col in df.columns


def test_split_into_seasons():
    time = np.array([0.0, 1.0, 2.0, 200.0, 201.0])
    labels = split_into_seasons(time, gap_days=100)
    # A >gap_days jump between index 2 and 3 must start a new season,
    # and points within the same cluster share a label.
    assert labels[3] != labels[2]
    assert labels[3] == labels[4]
    assert labels[0] == labels[0]  # first point keeps its own (original) label


def test_pipeline_runs_on_dataframe():
    df = run_pipeline_from_dataframe(make_lightcurve(with_event=True), CONFIG)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "obj1"
    # an injected event should at least trip the bump detector
    assert bool(df.iloc[0]["bump_flag"]) is True


def test_flat_lightcurve_is_not_a_candidate():
    df = run_pipeline_from_dataframe(make_lightcurve(with_event=False), CONFIG)
    assert bool(df.iloc[0]["is_candidate"]) is False
    assert df.iloc[0]["label"] == "no_event"


# ── Stage 5 labelling ────────────────────────────────────────────────────────

def make_anomalous_lightcurve(seed=0):
    """A PSPL event with a short spike on it, as a caustic crossing leaves.

    The spike is snapped to sampled epochs, since an anomaly injected into a
    gap is indistinguishable from no anomaly at all.
    """
    lc = make_lightcurve(with_event=True, seed=seed)
    time = lc["bjd"].to_numpy()
    spike_at = time[np.argmin(np.abs(time - 2459066.0))]
    lc["mag"] = lc["mag"] - 0.3 * np.exp(-0.5 * ((time - spike_at) / 1.0) ** 2)
    return lc


def make_periodic_lightcurve(period=3.0, seed=0):
    rng = np.random.default_rng(seed)
    time = np.concatenate([np.linspace(2459000, 2459120, 400),
                           np.linspace(2459300, 2459420, 400)])
    mag = 18.0 - 0.4 * np.sin(2 * np.pi * time / period) + rng.normal(0, 0.01, time.size)
    return pd.DataFrame({"bjd": time, "mag": mag,
                         "mag_err": np.full_like(time, 0.01), "name": "obj1"})


def test_label_is_always_one_of_the_declared_values():
    from aethra.schema import LABELS

    for lc in (make_lightcurve(with_event=False), make_lightcurve(with_event=True),
               make_anomalous_lightcurve(), make_periodic_lightcurve()):
        assert run_pipeline_from_dataframe(lc, CONFIG).iloc[0]["label"] in LABELS


def test_clean_event_is_pspl_like():
    row = run_pipeline_from_dataframe(make_lightcurve(with_event=True), CONFIG).iloc[0]

    assert row["label"] == "pspl_like"
    assert bool(row["is_candidate"]) is True
    assert bool(row["residual_significant"]) is False
    assert abs(row["tE_fit"] - 8.0) < 2.0


def test_an_anomaly_is_classified_not_rejected():
    """The whole point of the redesign. A spike on the peak makes the PSPL fit
    worse, and under the old chi2 gate that lost the object; now it is the
    thing that promotes it to non_pspl_candidate."""
    clean = run_pipeline_from_dataframe(make_lightcurve(with_event=True), CONFIG).iloc[0]
    anomalous = run_pipeline_from_dataframe(make_anomalous_lightcurve(), CONFIG).iloc[0]

    assert anomalous["chi2_red_pspl"] > clean["chi2_red_pspl"]
    assert bool(anomalous["is_candidate"]) is True
    assert anomalous["label"] == "non_pspl_candidate"
    assert bool(anomalous["residual_significant"]) is True
    assert bool(anomalous["residual_localized"]) is True


def test_periodic_variable_is_labelled_a_variable_star():
    row = run_pipeline_from_dataframe(make_periodic_lightcurve(), CONFIG).iloc[0]

    assert bool(row["hc_periodic"]) is True
    assert row["label"] == "variable_star"
    assert bool(row["is_variable_star"]) is True
    assert bool(row["is_candidate"]) is False


def test_is_candidate_is_an_alias_for_event_candidate():
    for lc in (make_lightcurve(with_event=False), make_lightcurve(with_event=True),
               make_anomalous_lightcurve(), make_periodic_lightcurve()):
        row = run_pipeline_from_dataframe(lc, CONFIG).iloc[0]
        assert bool(row["is_candidate"]) == bool(row["event_candidate"])


def make_long_event_lightcurve(n_seasons=8, tE=900.0, seed=3):
    """One very long event spread over many observing seasons."""
    rng = np.random.default_rng(seed)
    time = np.concatenate([np.linspace(0, 120, 360) + i * 300.0 + 2459000.0
                           for i in range(n_seasons)])
    t0 = time[0] + 0.5 * np.ptp(time)
    mag = 18.0 - 2.5 * np.log10(pspl_magnification(time, t0, 0.05, tE))
    return pd.DataFrame({"bjd": time, "mag": mag + rng.normal(0, 0.01, time.size),
                         "mag_err": np.full_like(time, 0.01), "name": "obj1"})


def test_recurrence_is_reported_but_does_not_veto():
    """A long event straddling seasonal gaps fragments into several excursions
    exactly as a variable does, so n_up cannot be a veto."""
    row = run_pipeline_from_dataframe(make_long_event_lightcurve(), CONFIG).iloc[0]

    assert row["n_up"] > 1
    assert bool(row["veto_recurrent"]) is True
    assert bool(row["is_variable_star"]) is False
    assert bool(row["is_candidate"]) is True


def test_a_long_event_is_not_mistaken_for_a_periodic_variable():
    """The periodicity mask is taken from the fitted tE, not from the detected
    excursion, which for a long event covers about one season. Masking only
    that left the wings behind and they fold at roughly the season spacing."""
    row = run_pipeline_from_dataframe(make_long_event_lightcurve(tE=700.0), CONFIG).iloc[0]

    assert bool(row["hc_periodic"]) is False
    assert row["label"] == "pspl_like"
    assert abs(row["tE_fit"] - 700.0) / 700.0 < 0.1


def test_detect_bump_finds_injected_bump():
    lc = make_lightcurve(with_event=True)
    flag, _, snr = detect_bump(
        lc["bjd"].to_numpy(), lc["mag"].to_numpy(), lc["mag_err"].to_numpy()
    )
    assert flag is True
    assert np.isfinite(snr)


def test_fit_pspl_recovers_parameters():
    lc = make_lightcurve(with_event=True)
    # restrict to the event season for a clean fit
    season = lc[lc["bjd"] < 2459200]
    result = fit_pspl(
        season["bjd"].to_numpy(), season["mag"].to_numpy(), season["mag_err"].to_numpy()
    )
    assert result is not None
    assert abs(result["t0_fit"] - 2459060.0) < 5.0
    assert result["tE_fit"] > 0


def test_plot_pspl_fit_writes_a_figure(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from aethra import plot_pspl_fit

    lc = make_lightcurve(with_event=True)
    output_path = tmp_path / "pspl-fit.png"
    fig, axes, fit_result = plot_pspl_fit(
        lc["bjd"], lc["mag"], lc["mag_err"], save_path=output_path
    )

    assert output_path.exists()
    assert len(axes) == 2
    assert fit_result["chi2_red_pspl"] >= 0
    fig.clf()


def test_load_and_run_accepts_dataframe():
    df = load_and_run(make_lightcurve(), CONFIG)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == OUTPUT_COLUMNS


def test_load_and_run_from_csv(tmp_path):
    lc = make_lightcurve()
    path = tmp_path / "data.csv"
    lc.to_csv(path, index=False)
    cfg = {**CONFIG, "sep": ",", "header": 0, "columns": None}
    df = load_and_run(str(path), cfg)
    assert len(df) == 1


def test_missing_column_raises():
    bad = make_lightcurve().rename(columns={"mag": "brightness"})
    with pytest.raises(ValueError):
        run_pipeline_from_dataframe(bad, CONFIG)


# ── YAML configuration ────────────────────────────────────────────────────────

def test_load_config_reads_yaml(tmp_path):
    from aethra import load_config

    p = tmp_path / "config.yaml"
    p.write_text(
        "time_col: bjd\n"
        "mag_col: mag\n"
        "err_col: mag_err\n"
        "group_col: name\n"
        "season_gap_days: 120\n"
    )
    cfg = load_config(str(p))
    assert cfg["time_col"] == "bjd"
    assert cfg["group_col"] == "name"
    assert cfg["season_gap_days"] == 120


def test_load_config_null_becomes_none(tmp_path):
    from aethra import load_config

    p = tmp_path / "config.yaml"
    p.write_text("time_col: bjd\nmag_col: mag\nerr_col: mag_err\ngroup_col: null\n")
    cfg = load_config(str(p))
    assert cfg["group_col"] is None


def test_load_config_rejects_non_mapping(tmp_path):
    from aethra import load_config

    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_run_from_yaml_config(tmp_path):
    from aethra import load_config

    lc = make_lightcurve()
    data_path = tmp_path / "data.csv"
    lc.to_csv(data_path, index=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "time_col: bjd\n"
        "mag_col: mag\n"
        "err_col: mag_err\n"
        "group_col: name\n"
        "sep: ','\n"
        "header: 0\n"
    )
    cfg = load_config(str(cfg_path))
    df = load_and_run(str(data_path), cfg)
    assert len(df) == 1


def test_cli_config_with_flag_override(tmp_path):
    from aethra.cli import build_config, build_parser

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("time_col: bjd\nmag_col: mag\nerr_col: mag_err\nmin_points: 10\n")

    args = build_parser().parse_args(
        ["data.csv", "--config", str(cfg_path), "--min-points", "25"]
    )
    config = build_config(args)
    # CLI flag wins over the YAML value
    assert config["min_points"] == 25
    # untouched YAML keys survive
    assert config["time_col"] == "bjd"


# ── roman_variable FITS loader ───────────────────────────────────────────────

def _write_roman_variable_fixture(tmp_path):
    time_dir = tmp_path / "top_level"
    time_dir.mkdir()
    times = {
        "F087": np.array([1.0, 2.0]),
        "F146": np.array([1.1, 1.2, 1.3]),
        "F213": np.array([1.05, 2.05]),
    }
    filenames = {
        "F087": "roman_times_longcadence.npy",
        "F146": "roman_times_shortcadence.npy",
        "F213": "roman_times_longcadence2.npy",
    }
    for filt, values in times.items():
        np.save(time_dir / filenames[filt], values)

    primary = fits.PrimaryHDU()
    primary.header["NAME"] = "fixture-object"
    hdus = [primary]
    for filt, values in times.items():
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="mag",
                        format="D",
                        array=np.arange(len(values)) + 18.0,
                    ),
                    fits.Column(
                        name="mag_error",
                        format="D",
                        array=np.full(len(values), 0.01),
                    ),
                ],
                name=filt,
            )
        )
    fits_path = tmp_path / "fixture.fits"
    fits.HDUList(hdus).writeto(fits_path)
    return fits_path, time_dir


def test_load_roman_variable_reads_hdus_and_external_times(tmp_path):
    fits_path, time_dir = _write_roman_variable_fixture(tmp_path)

    df = load_roman_variable(fits_path, time_dir=time_dir)

    assert list(df.columns) == ["bjd", "mag", "mag_err", "filt", "name"]
    assert len(df) == 7
    assert df["name"].unique().tolist() == ["fixture-object"]
    assert df.groupby("filt").size().to_dict() == {"F087": 2, "F146": 3, "F213": 2}
    assert df["bjd"].is_monotonic_increasing


def test_load_roman_variable_can_select_filters(tmp_path):
    fits_path, time_dir = _write_roman_variable_fixture(tmp_path)

    df = load_roman_variable(fits_path, time_dir=time_dir, filters="F146")

    assert len(df) == 3
    assert df["filt"].unique().tolist() == ["F146"]


def test_a_wandering_baseline_is_a_variable_star():
    """The renormalization factor is also a variable-star statistic, and it
    says what the periodicity test cannot: the curve away from the event is
    not constant either. That needs no period, which is why it catches the
    semi-regular variables that fold badly."""
    rng = np.random.default_rng(11)
    time = np.concatenate([np.linspace(0, 120, 360) + i * 300.0 + 2459000.0
                           for i in range(4)])
    # A bump big enough to detect, on a baseline that is nowhere near constant.
    mag = (18.0
           - 1.5 * np.exp(-0.5 * ((time - time[600]) / 25.0) ** 2)
           + 0.3 * rng.normal(0, 1, time.size))
    lc = pd.DataFrame({"bjd": time, "mag": mag,
                       "mag_err": np.full_like(time, 0.001), "name": "obj1"})

    row = run_pipeline_from_dataframe(lc, CONFIG).iloc[0]

    assert row["error_renorm"] > 1000.0
    assert bool(row["veto_baseline_variable"]) is True
    assert row["label"] == "variable_star"
    assert bool(row["is_candidate"]) is False


def test_the_baseline_criterion_is_configurable():
    lc = make_lightcurve(with_event=True)
    strict = run_pipeline_from_dataframe(lc, {**CONFIG, "max_error_renorm": 0.5}).iloc[0]

    assert bool(strict["veto_baseline_variable"]) is True
    assert strict["label"] == "variable_star"
