"""Tests for the FFT matched-filter search for PSPL starting values."""

import numpy as np
import pytest

from aethra import pspl_magnification
from aethra.pspl_seed import (
    DEFAULT_TEFF_GRID,
    DEFAULT_U0_GRID,
    pspl_excess_magnification,
    pspl_fft_seed,
)


def survey_times(n_seasons=4, season_days=120, gap_days=250, per_day=4, seed=0):
    rng = np.random.default_rng(seed)
    chunks = []
    start = 2459000.0
    for _ in range(n_seasons):
        chunks.append(np.sort(rng.uniform(start, start + season_days, season_days * per_day)))
        start += season_days + gap_days
    return np.concatenate(chunks)


def event_flux(t0, tE, u0, noise=0.005, seed=0, time=None, blend=0.0):
    time = survey_times() if time is None else time
    rng = np.random.default_rng(seed + 1)
    source = 10 ** (-0.4 * 18.0)
    flux = source * pspl_magnification(time, t0, u0, tE) + blend * source
    flux_err = np.full_like(time, noise * source)
    return time, flux + rng.normal(0, 1, time.size) * flux_err, flux_err


# ── the template ─────────────────────────────────────────────────────────────

def test_excess_magnification_agrees_with_the_direct_form():
    """Where cancellation is not an issue the two must be the same function."""
    u0, teff = 0.3, 10.0
    tE = teff / u0
    lag = np.linspace(-30.0, 30.0, 201)

    excess = pspl_excess_magnification(lag, u0, teff)
    direct = pspl_magnification(lag, 0.0, u0, tE) - 1.0

    assert excess == pytest.approx(direct, rel=1e-9)


def test_excess_magnification_survives_the_far_wings():
    """The reason for the rationalised form: far out, A - 1 computed directly
    loses its significant digits to cancellation, and the matched filter is
    weighing precisely that difference."""
    u0, teff = 0.3, 10.0
    lag = np.array([1e5, 1e6, 1e7])

    excess = pspl_excess_magnification(lag, u0, teff)

    assert np.all(excess > 0)
    assert np.all(np.isfinite(excess))
    # A - 1 falls off as u^-4 in the wings, so each decade in lag costs 4.
    ratio = excess[0] / excess[1]
    assert ratio == pytest.approx(1e4, rel=0.05)


def test_excess_magnification_rejects_nonpositive_shape_parameters():
    with pytest.raises(ValueError):
        pspl_excess_magnification([0.0], 0.0, 10.0)
    with pytest.raises(ValueError):
        pspl_excess_magnification([0.0], 0.3, -1.0)


# ── the search ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tE,u0", [(5.0, 0.1), (25.0, 0.3), (80.0, 0.15), (200.0, 0.5)])
def test_seed_lands_near_the_injected_event(tE, u0):
    """The seed is coarse by construction — the bank is a handful of widths —
    so it is required to land in the basin, not on the answer."""
    time = survey_times(n_seasons=6)
    t0 = float(time[len(time) // 2 - 100])
    time, flux, flux_err = event_flux(t0, tE, u0, time=time)

    seed = pspl_fft_seed(time, flux, flux_err)

    assert np.isfinite(seed["t0"])
    assert abs(seed["t0"] - t0) < max(5.0, 0.5 * tE)
    assert 0.25 * tE < seed["tE"] < 4.0 * tE
    assert seed["delta_chi2"] > 0


def test_seed_is_parameterized_by_teff():
    """tE is reported, but the bank is gridded in teff = u0 * tE, which is the
    width the bump actually shows. Every seed must respect that identity."""
    time, flux, flux_err = event_flux(2459060.0, 25.0, 0.3)
    seed = pspl_fft_seed(time, flux, flux_err)

    assert seed["teff"] == pytest.approx(seed["u0"] * seed["tE"])
    assert seed["teff"] in DEFAULT_TEFF_GRID
    assert seed["u0"] in DEFAULT_U0_GRID


def test_seed_finds_nothing_in_a_flat_curve():
    time = survey_times()
    rng = np.random.default_rng(2)
    source = 10 ** (-0.4 * 18.0)
    flux_err = np.full_like(time, 0.005 * source)
    flux = source + rng.normal(0, 1, time.size) * flux_err

    seed = pspl_fft_seed(time, flux, flux_err)

    # Noise still correlates weakly with some template, so the test is that the
    # improvement is negligible against the baseline, not that it is zero.
    assert seed["delta_chi2"] < 0.05 * seed["null_chi2"]


def test_seed_ignores_dimmings():
    """``positive_source`` exists so the filter answers about brightenings.

    A reflected event is not scored at zero — with a trough present the
    weighted mean sits below the baseline, so everything outside the trough is
    an excess a broad template can partly absorb. What must not happen is the
    seed pointing *at* the trough.
    """
    time = survey_times()
    t0 = float(time[len(time) // 2 - 100])
    _, flux, flux_err = event_flux(t0, 25.0, 0.3, time=time)
    source = 10 ** (-0.4 * 18.0)
    dimmed = 2.0 * source - flux  # same shape, reflected about the baseline

    bright = pspl_fft_seed(time, flux, flux_err)
    dim = pspl_fft_seed(time, dimmed, flux_err)

    assert bright["delta_chi2"] > 0.9 * bright["null_chi2"]
    assert dim["delta_chi2"] < 0.2 * dim["null_chi2"]
    assert abs(dim["t0"] - t0) > 10 * 25.0


def test_top_k_returns_ranked_seeds():
    time, flux, flux_err = event_flux(2459060.0, 25.0, 0.3)
    seed = pspl_fft_seed(time, flux, flux_err, top_k=5)

    assert seed["seeds"].shape == (5, 3)
    assert seed["seeds"][0, 0] == seed["t0"]
    assert seed["seeds"][0, 1] == pytest.approx(seed["tE"])


def test_seed_handles_degenerate_input():
    for args in ([[], [], []], [[1.0], [1.0], [1.0]]):
        seed = pspl_fft_seed(*args)
        assert len(seed["seeds"]) == 0
        assert not np.isfinite(seed["t0"])


def test_seed_refuses_an_oversized_grid():
    """The grid spacing is the narrowest template over ``samples_per_teff``, so
    a very narrow template against a long baseline would allocate without
    bound. It declines rather than trying."""
    time, flux, flux_err = event_flux(2459060.0, 25.0, 0.3)
    seed = pspl_fft_seed(time, flux, flux_err, max_grid_points=100)

    assert len(seed["seeds"]) == 0


def test_seed_rejects_an_empty_template_bank():
    time, flux, flux_err = event_flux(2459060.0, 25.0, 0.3)
    assert len(pspl_fft_seed(time, flux, flux_err, u0_grid=[])["seeds"]) == 0
    assert len(pspl_fft_seed(time, flux, flux_err, teff_grid=[-1.0])["seeds"]) == 0
