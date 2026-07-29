"""Tests for error renormalization, full-curve PSPL fitting and residual structure."""

import numpy as np
import pytest

from aethra import pspl_magnification
from aethra.coherence import coherent_peak_scan
from aethra.pspl import fit_pspl, fit_pspl_full, renormalize_errors, residual_structure


def multi_season_times(n_seasons=4, season_days=120, gap_days=250, per_day=4, seed=0):
    rng = np.random.default_rng(seed)
    chunks = []
    start = 2459000.0
    for _ in range(n_seasons):
        chunks.append(np.sort(rng.uniform(start, start + season_days, season_days * per_day)))
        start += season_days + gap_days
    return np.concatenate(chunks)


def observed_time_inside_a_season(time):
    """An epoch that is actually observed; ``median(time)`` falls in a gap."""
    return float(time[len(time) // 2 - 100])


def make_event(tE=25.0, u0=0.15, noise=0.005, seed=0, anomaly_mag=0.0,
               anomaly_at=None, anomaly_width=1.0, err_scale=1.0, n_seasons=4):
    """A PSPL curve, optionally with a narrow anomaly superimposed.

    ``anomaly_at`` is snapped to an observed epoch: placed by arithmetic it
    lands in a seasonal gap and injects nothing at all.
    """
    time = multi_season_times(n_seasons=n_seasons)
    t0 = observed_time_inside_a_season(time)
    mag = 18.0 - 2.5 * np.log10(pspl_magnification(time, t0, u0, tE))

    if anomaly_mag:
        target = t0 if anomaly_at is None else t0 + anomaly_at
        snapped = time[np.argmin(np.abs(time - target))]
        mag = mag - anomaly_mag * np.exp(-0.5 * ((time - snapped) / anomaly_width) ** 2)

    rng = np.random.default_rng(seed + 1)
    mag = mag + rng.normal(0, noise, size=mag.size)
    # err_scale < 1 quotes errors smaller than the scatter actually present.
    return time, mag, np.full_like(time, noise * err_scale), t0


def analyse(time, mag, err):
    """The Stage 1 -> 3 -> 4 chain, as the pipeline will run it."""
    scan = coherent_peak_scan(time, mag, err)
    renorm = renormalize_errors(time, mag, err, scan["peak_time"],
                                scan["main_duration_days"])
    fit = fit_pspl_full(time, mag, err, scan["peak_time"], scan["main_duration_days"],
                        error_renorm=renorm["error_renorm"])
    structure = residual_structure(fit["residual_time"], fit["residual_flux"],
                                   fit["t0_fit"], fit["tE_fit"])
    return scan, renorm, fit, structure


# ── Stage 1: error renormalization ───────────────────────────────────────────

def test_renormalization_is_a_no_op_on_honest_errors():
    time, mag, err, t0 = make_event()
    scan = coherent_peak_scan(time, mag, err)
    result = renormalize_errors(time, mag, err, scan["peak_time"],
                                scan["main_duration_days"])

    assert result["baseline_chi2_red"] == pytest.approx(1.0, abs=0.15)
    assert result["error_renorm"] == pytest.approx(1.0, abs=0.15)


def test_renormalization_recovers_an_understated_error_scale():
    """Errors quoted at half the true scatter inflate every chi2 fourfold."""
    time, mag, err, t0 = make_event(err_scale=0.5)
    scan = coherent_peak_scan(time, mag, err)
    renorm = renormalize_errors(time, mag, err, scan["peak_time"],
                                scan["main_duration_days"])
    fit = fit_pspl_full(time, mag, err, scan["peak_time"], scan["main_duration_days"],
                        error_renorm=renorm["error_renorm"])

    assert renorm["baseline_chi2_red"] == pytest.approx(4.0, rel=0.25)
    # Uncorrected this event fails a chi2 < 2.5 gate on error bars alone.
    assert fit["chi2_red_pspl"] > 2.5
    assert fit["chi2_red_pspl_renorm"] == pytest.approx(1.0, abs=0.3)


def test_renormalization_does_not_correct_conservative_errors():
    """``k < 1`` is not evidence the errors are too large, so it is clamped:
    dividing by it would manufacture a bad fit from cautious error bars."""
    time, mag, err, t0 = make_event(err_scale=2.0)
    scan = coherent_peak_scan(time, mag, err)
    result = renormalize_errors(time, mag, err, scan["peak_time"],
                                scan["main_duration_days"])

    assert result["baseline_chi2_red"] < 0.5
    assert result["error_renorm"] == 1.0


def test_renormalization_excludes_the_event_window():
    """Measured over the event itself, the baseline scatter is the event."""
    time, mag, err, t0 = make_event(tE=25.0, u0=0.05)
    scan = coherent_peak_scan(time, mag, err)

    masked = renormalize_errors(time, mag, err, scan["peak_time"],
                                scan["main_duration_days"])
    unmasked = renormalize_errors(time, mag, err)

    assert unmasked["baseline_chi2_red"] > 10 * masked["baseline_chi2_red"]
    assert masked["baseline_chi2_red"] == pytest.approx(1.0, abs=0.2)


def test_renormalization_reports_no_correction_without_enough_points():
    result = renormalize_errors([1.0, 2.0], [18.0, 18.0], [0.01, 0.01])
    assert result["error_renorm"] == 1.0
    assert not np.isfinite(result["baseline_chi2_red"])


# ── Stage 3: full-curve fitting ──────────────────────────────────────────────

def test_full_fit_recovers_injected_parameters():
    time, mag, err, t0 = make_event(tE=25.0, u0=0.15)
    scan, _, fit, _ = analyse(time, mag, err)

    assert fit["t0_fit"] == pytest.approx(t0, abs=1.0)
    assert fit["tE_fit"] == pytest.approx(25.0, rel=0.1)
    assert fit["u0_fit"] == pytest.approx(0.15, rel=0.15)
    assert fit["frac_explained"] > 0.9


@pytest.mark.parametrize("tE", [15.0, 60.0, 200.0])
def test_full_fit_holds_up_where_the_season_scoped_fit_does_not(tE):
    """A season truncates the wings that constrain tE. The season-scoped fit
    is given one season, which is what the pipeline does today; the full fit
    is given the whole curve and must do at least as well on tE."""
    time, mag, err, t0 = make_event(tE=tE, u0=0.2, n_seasons=6)
    season = np.abs(time - t0) < 60.0

    scan, _, full, _ = analyse(time, mag, err)
    scoped = fit_pspl(time[season], mag[season], err[season])

    assert full["tE_fit"] == pytest.approx(tE, rel=0.2)
    if scoped is not None and tE > 60.0:
        full_error = abs(full["tE_fit"] - tE)
        scoped_error = abs(scoped["tE_fit"] - tE)
        assert full_error <= scoped_error


def test_full_fit_returns_none_for_too_little_data():
    assert fit_pspl_full([1.0, 2.0], [18.0, 18.0], [0.01, 0.01]) is None


def test_frac_explained_is_near_zero_for_a_flat_curve():
    time = multi_season_times()
    rng = np.random.default_rng(4)
    mag = 18.0 + rng.normal(0, 0.01, size=time.size)
    fit = fit_pspl_full(time, mag, np.full_like(time, 0.01))

    assert fit is not None
    assert fit["frac_explained"] < 0.1


# ── Stage 4: residual structure ──────────────────────────────────────────────

def test_a_clean_fit_leaves_no_significant_residual():
    time, mag, err, t0 = make_event()
    _, _, _, structure = analyse(time, mag, err)

    assert not structure["residual_significant"]
    assert not structure["residual_localized"]


def test_proximity_alone_does_not_make_a_residual_localized():
    """Noise residuals land within an einstein time of t0 by chance, so
    ``residual_localized`` must require significance as well as proximity."""
    time, mag, err, t0 = make_event(seed=7)
    _, _, fit, _ = analyse(time, mag, err)

    lenient = residual_structure(fit["residual_time"], fit["residual_flux"],
                                 fit["t0_fit"], fit["tE_fit"], min_peak_score=0.0)
    strict = residual_structure(fit["residual_time"], fit["residual_flux"],
                                fit["t0_fit"], fit["tE_fit"])

    assert lenient["residual_offset_tE"] < 2.0
    assert lenient["residual_localized"]
    assert not strict["residual_localized"]


def test_an_anomaly_at_the_peak_is_localized_despite_failing_the_chi2_gate():
    """The case the pipeline currently discards: a planetary deviation makes
    the PSPL fit bad, and a chi2 gate reads bad-fit as not-an-event."""
    time, mag, err, t0 = make_event(anomaly_mag=0.25, anomaly_at=8.0)
    scan, _, fit, structure = analyse(time, mag, err)

    assert fit["chi2_red_pspl"] > 2.5      # fails today's gate
    assert scan["peak_score"] > 40         # but detection is unambiguous
    assert structure["residual_significant"]
    assert structure["residual_localized"]
    assert structure["residual_sign"] == 1


def test_structure_far_from_the_peak_is_significant_but_not_localized():
    """Same bad chi2, opposite meaning: systematics or an unmodelled variable."""
    time, mag, err, t0 = make_event(anomaly_mag=0.25, anomaly_at=700.0)
    _, _, fit, structure = analyse(time, mag, err)

    assert fit["chi2_red_pspl"] > 2.5
    assert structure["residual_significant"]
    assert not structure["residual_localized"]
    assert structure["residual_offset_tE"] > 2.0


def test_a_dimming_anomaly_is_found_with_a_negative_sign():
    time, mag, err, t0 = make_event(anomaly_mag=-0.15, anomaly_at=6.0)
    _, _, fit, structure = analyse(time, mag, err)

    assert structure["residual_significant"]
    assert structure["residual_localized"]
    assert structure["residual_sign"] == -1


def test_residual_structure_handles_an_unfittable_curve():
    structure = residual_structure([1.0, 2.0], [0.0, 0.0])
    assert not structure["residual_significant"]
    assert not structure["residual_localized"]
