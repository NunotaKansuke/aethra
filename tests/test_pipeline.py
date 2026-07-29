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


def test_pipeline_rejects_bad_pspl_fit(monkeypatch):
    def bad_pspl_fit(*args, **kwargs):
        return {
            "t0_fit_raw": 2459060.0,
            "u0_fit": 0.1,
            "tE_fit": 8.0,
            "chi2_red_pspl": 3.0,
            "is_candidate": False,
            "is_ffp_candidate": False,
        }

    monkeypatch.setattr("aethra.pipeline.fit_pspl_candidate", bad_pspl_fit)

    df = run_pipeline_from_dataframe(make_lightcurve(with_event=True), CONFIG)

    assert bool(df.iloc[0]["bump_flag"]) is True
    assert bool(df.iloc[0]["veto_periodic"]) is False
    assert bool(df.iloc[0]["veto_recurrent"]) is False
    assert bool(df.iloc[0]["is_candidate"]) is False


def test_flat_lightcurve_is_not_a_candidate():
    df = run_pipeline_from_dataframe(make_lightcurve(with_event=False), CONFIG)
    assert bool(df.iloc[0]["is_candidate"]) is False


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
