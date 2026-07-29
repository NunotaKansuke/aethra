"""Tests for season-free coherent-excursion detection and full-curve periodicity."""

import numpy as np
import pytest

from aethra import pspl_magnification
from aethra.coherence import (
    coherent_excursions,
    coherent_peak_scan,
    daily_median_bins,
    smooth_multiscale,
)
from aethra.variability import analyze_periodicity, periodicity_with_event_masked


def multi_season_times(n_seasons=4, season_days=120, gap_days=250, per_day=4, seed=0):
    """Roman-like sampling: dense seasons separated by long gaps."""
    rng = np.random.default_rng(seed)
    chunks = []
    start = 2459000.0
    for _ in range(n_seasons):
        n = season_days * per_day
        chunks.append(np.sort(rng.uniform(start, start + season_days, n)))
        start += season_days + gap_days
    return np.concatenate(chunks)


def observed_time_inside_a_season(time):
    """An epoch that is actually observed, not one sitting in a seasonal gap.

    ``median(time)`` is not safe here: with dense seasons and long gaps it
    falls in a gap, and an event placed there is genuinely unobservable.
    """
    return float(time[len(time) // 2 - 100])


def event_curve(t0, tE, u0=0.1, baseline=18.0, noise=0.01, seed=0, time=None):
    time = multi_season_times(seed=seed) if time is None else time
    rng = np.random.default_rng(seed + 1)
    mag = baseline - 2.5 * np.log10(pspl_magnification(time, t0, u0, tE))
    mag = mag + rng.normal(0, noise, size=mag.size)
    return time, mag, np.full_like(time, noise)


def sinusoid_curve(period, amplitude=0.3, baseline=18.0, noise=0.01, seed=0):
    time = multi_season_times(seed=seed)
    rng = np.random.default_rng(seed + 1)
    mag = baseline + amplitude * np.sin(2 * np.pi * time / period)
    mag = mag + rng.normal(0, noise, size=mag.size)
    return time, mag, np.full_like(time, noise)


# ── binning and smoothing primitives ─────────────────────────────────────────

def test_daily_median_bins_only_keeps_observed_nights():
    time = np.array([0.1, 0.2, 0.9, 5.3, 5.4])
    values = np.array([1.0, 3.0, 2.0, 10.0, 20.0])
    binned_time, binned_value, counts = daily_median_bins(time, values)

    # Two observed nights (day 0 and day 5); the empty days 1-4 create no bins.
    assert len(binned_time) == 2
    assert counts.tolist() == [3, 2]
    assert binned_value[0] == pytest.approx(2.0)
    assert binned_value[1] == pytest.approx(15.0)


def test_daily_median_bins_handles_empty_input():
    binned_time, binned_value, counts = daily_median_bins([], [])
    assert len(binned_time) == len(binned_value) == len(counts) == 0


def test_smooth_multiscale_prefers_the_width_matching_the_feature():
    time = np.arange(0.0, 200.0)
    values = np.zeros_like(time)
    values[95:105] = 10.0  # a ~10 day wide block

    best = smooth_multiscale(time, values)

    assert best["smoothed"] is not None
    assert np.isfinite(best["peak_score"])
    # A 30 day window would dilute a 10 day feature.
    assert best["half_width"] <= 7.0


def test_smooth_multiscale_reports_no_peak_without_enough_nights():
    best = smooth_multiscale(np.array([0.0, 1.0]), np.array([1.0, 1.0]), min_nights=5)
    assert best["smoothed"] is None
    assert not np.isfinite(best["peak_score"])


def test_coherent_excursions_merges_across_a_seasonal_gap():
    # One excursion sampled either side of a 15 day gap in coverage.
    time = np.array([0.0, 1.0, 2.0, 17.0, 18.0])
    smoothed = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

    assert len(coherent_excursions(time, smoothed, max_join_days=20.0)) == 1
    assert len(coherent_excursions(time, smoothed, max_join_days=5.0)) == 2


def test_coherent_excursions_ignores_sub_threshold_structure():
    time = np.arange(10.0)
    assert coherent_excursions(time, np.full(10, 1.0), sigma=3.0) == []


# ── the detection statistic ──────────────────────────────────────────────────

def test_scan_finds_an_injected_event():
    time, mag, err = event_curve(t0=2459060.0, tE=15.0)
    scan = coherent_peak_scan(time, mag, err)

    assert scan["peak_score"] > 40
    assert scan["n_up"] == 1
    assert abs(scan["peak_time"] - 2459060.0) < 10.0
    # magnification is one-sided
    assert scan["pos_ratio"] > 0.8


def test_scan_is_quiet_on_a_flat_curve():
    time = multi_season_times()
    rng = np.random.default_rng(3)
    mag = 18.0 + rng.normal(0, 0.01, size=time.size)
    scan = coherent_peak_scan(time, mag, np.full_like(time, 0.01))

    assert not (scan["peak_score"] > 40)
    assert scan["n_up"] == 0


@pytest.mark.parametrize("tE", [15.0, 60.0, 150.0, 400.0, 700.0])
def test_long_events_are_detected_not_vetoed(tE):
    """The failure mode the season-based recurrent veto exhibits: all 19 of its
    firings on the roman_simu sample were long events, not variable stars
    (median true tE 73 d against 12 d for the rest).

    The season-free scan must detect them at every duration. The survey has to
    outlast the event for this to be answerable at all: a 700 d event inside a
    1230 d window magnifies every observed night, leaving no baseline to
    measure against.
    """
    time = multi_season_times(n_seasons=8)
    t0 = observed_time_inside_a_season(time)
    time, mag, err = event_curve(t0=t0, tE=tE, u0=0.2, time=time)

    assert coherent_peak_scan(time, mag, err)["peak_score"] > 40


@pytest.mark.parametrize("tE", [15.0, 60.0, 150.0])
def test_an_event_within_a_season_is_one_excursion(tE):
    time = multi_season_times()
    t0 = observed_time_inside_a_season(time)
    time, mag, err = event_curve(t0=t0, tE=tE, u0=0.2, time=time)

    assert coherent_peak_scan(time, mag, err)["n_up"] == 1


def test_max_join_days_governs_merging_across_seasonal_gaps():
    """A known limit, pinned deliberately. ``max_join_days`` defaults to 20 d,
    below a seasonal gap, so an event bright across several seasons is counted
    once per season rather than once: on roman_simu only 60.5% of events with
    tE > 50 d score ``n_up == 1``. That still beats the season-based veto,
    which fired on 19 long events and caught no variable at all, and raising
    the default would merge the recurrent bumps of a variable star, which is
    the behaviour the count exists to expose.

    It takes a survey several times the event duration to see this: with too
    few seasons the baseline is measured from the event's own wings, the noise
    scale inflates, and only the single brightest season clears the threshold.
    """
    time = multi_season_times(n_seasons=8)
    t0 = observed_time_inside_a_season(time)
    time, mag, err = event_curve(t0=t0, tE=300.0, u0=0.2, time=time)

    assert coherent_peak_scan(time, mag, err)["n_up"] > 1
    assert coherent_peak_scan(time, mag, err, max_join_days=400.0)["n_up"] == 1


def test_refinement_takes_effect_on_the_reported_curve():
    """The reported statistic must come from the baseline the last pass used.

    An event covering most of the window inflates the noise scale with its own
    wings, so masking it and re-measuring roughly doubles the peak amplitude
    here. A refinement pass that updates the baseline but reports the curve
    smoothed against the previous one leaves the statistic untouched while
    silently changing the night counts derived from it.
    """
    time = multi_season_times(n_seasons=4)
    t0 = observed_time_inside_a_season(time)
    time, mag, err = event_curve(t0=t0, tE=700.0, u0=0.2, time=time)

    plain = coherent_peak_scan(time, mag, err, n_refine=0)
    refined = coherent_peak_scan(time, mag, err, n_refine=1)
    twice = coherent_peak_scan(time, mag, err, n_refine=2)

    assert refined["peak_sigma"] > 1.5 * plain["peak_sigma"]
    # A second pass finds the same baseline and stops.
    assert twice["peak_sigma"] == pytest.approx(refined["peak_sigma"])


def test_an_event_falling_in_a_seasonal_gap_is_not_detectable():
    """Not a defect: if the peak is never observed there is nothing to find.
    Pinned so a future baseline change cannot quietly start inventing it."""
    time = multi_season_times()
    gap_t0 = float(np.median(time))  # lands between two seasons
    assert np.min(np.abs(time - gap_t0)) > 50.0

    time, mag, err = event_curve(t0=gap_t0, tE=30.0, u0=0.2, time=time)
    scan = coherent_peak_scan(time, mag, err)

    assert scan["n_up"] == 0


def test_scan_separates_an_event_from_a_sinusoid():
    _, event_mag, err = event_curve(t0=2459060.0, tE=15.0)
    time = multi_season_times()
    _, variable_mag, _ = sinusoid_curve(period=60.0)

    event = coherent_peak_scan(time, event_mag, err)
    variable = coherent_peak_scan(time, variable_mag, err)

    assert event["peak_score"] > variable["peak_score"]
    # A sinusoid spends as much time below baseline as above it.
    assert variable["pos_ratio"] < 0.75
    assert event["pos_ratio"] > variable["pos_ratio"]


def test_scan_returns_empty_result_for_too_little_data():
    scan = coherent_peak_scan([1.0, 2.0], [18.0, 18.0], [0.01, 0.01])
    assert not np.isfinite(scan["peak_score"])
    assert scan["n_up"] == 0


def test_scan_drops_non_finite_points():
    time, mag, err = event_curve(t0=2459060.0, tE=15.0)
    mag = mag.copy()
    mag[::50] = np.nan
    err = err.copy()
    err[1::50] = 0.0  # non-positive errors are dropped too

    scan = coherent_peak_scan(time, mag, err)
    assert scan["peak_score"] > 40


# ── full-curve periodicity ───────────────────────────────────────────────────

def test_analyze_periodicity_recovers_a_long_period():
    """The per-season test caps the period at min(50, 0.7 * season length) and
    so cannot reach long-period variables at all."""
    time, mag, err = sinusoid_curve(period=180.0)
    flux = 10 ** (-0.4 * mag)

    result = analyze_periodicity(time, flux, err)

    assert result["valid"]
    assert result["period"] == pytest.approx(180.0, rel=0.1)
    assert result["high_confidence_periodic"]
    assert result["variance_reduction"] > 0.3


def test_analyze_periodicity_is_quiet_on_noise():
    time = multi_season_times()
    rng = np.random.default_rng(5)
    flux = 1.0 + rng.normal(0, 0.01, size=time.size)

    result = analyze_periodicity(time, flux)
    assert not result["high_confidence_periodic"]


def test_masking_resolves_an_event_mistaken_for_a_period():
    """A single smooth bump is itself a low-FAP signal: run on the raw curve,
    the periodogram latches onto the event and reports a period comparable to
    the observed span. On the roman_simu sample this mislabels about a quarter
    of real events; masking the event window first brings it to a few percent.
    """
    time, mag, err = event_curve(t0=2459060.0, tE=15.0)
    flux = 10 ** (-0.4 * mag)
    scan = coherent_peak_scan(time, mag, err)

    unmasked = analyze_periodicity(time, flux, err)
    masked = periodicity_with_event_masked(
        time, flux, err,
        peak_time=scan["peak_time"], duration_days=scan["main_duration_days"],
    )

    # the "period" it finds is just the event itself, at a fraction of the span
    assert unmasked["high_confidence_periodic"]
    assert unmasked["cycles"] < 10
    assert not masked["high_confidence_periodic"]


def test_analyze_periodicity_rejects_unknown_options():
    with pytest.raises(ValueError, match="Unknown periodicity option"):
        analyze_periodicity([1.0], [1.0], max_perod=10.0)


def test_analyze_periodicity_reports_short_span():
    time = np.linspace(0.0, 1.0, 40)
    result = analyze_periodicity(time, np.ones_like(time))
    assert not result["valid"]
    assert result["reason"] == "observed span too short"


def test_masking_the_event_does_not_hide_a_real_variable():
    time, mag, err = sinusoid_curve(period=180.0)
    flux = 10 ** (-0.4 * mag)
    scan = coherent_peak_scan(time, mag, err)

    result = periodicity_with_event_masked(
        time, flux, err,
        peak_time=scan["peak_time"], duration_days=scan["main_duration_days"],
    )
    assert result["high_confidence_periodic"]


def test_masking_abstains_when_the_event_covers_the_baseline():
    """An event window that swallows the whole curve leaves nothing to test
    against. Falling back to the unmasked curve would report the event's own
    rise and fall as a period, which is how long microlensing events were
    being labelled variable stars."""
    time, mag, err = sinusoid_curve(period=180.0)
    flux = 10 ** (-0.4 * mag)

    result = periodicity_with_event_masked(
        time, flux, err, peak_time=float(np.median(time)),
        duration_days=1e6,
    )

    assert result["valid"] is False
    assert result["reason"] == "event_covers_baseline"
    assert result["high_confidence_periodic"] is False


def test_masking_is_a_no_op_without_an_event_window():
    time, mag, err = sinusoid_curve(period=180.0)
    flux = 10 ** (-0.4 * mag)

    masked = periodicity_with_event_masked(time, flux, err)
    plain = analyze_periodicity(time, flux, err)

    assert masked["period"] == pytest.approx(plain["period"])
