"""FFT matched-filter search for PSPL starting values.

Fitting PSPL directly is a three-parameter non-linear problem with a badly
behaved surface: tE and u0 trade off against each other through the peak
amplitude, so a local optimiser settles into whichever minimum it started
nearest. The usual answer is a multi-start over u0, which multiplies the cost
of an already expensive fit without searching t0 globally at all.

This module removes both problems by exploiting the structure of the model.
Written as ``flux = Fs * A(t; t0, u0, teff) + Fb``, the two flux parameters are
linear and can be profiled out in closed form, and t0 enters only as a
translation — so the profiled chi2 at *every* t0 on a grid is one
cross-correlation, which is one FFT. Only ``(u0, teff)`` needs an explicit
grid, and that grid is small.

The parameterisation is the other half of it. The scan works in
``teff = u0 * tE`` rather than tE, because teff is the observable width of the
bump while u0 sets its height; those two are far less degenerate than tE and u0,
so a coarse grid covers the space that a multi-start over u0 was groping at.

The approach and its formulation are taken from ``jacscanomaly.pspl_fft``
(``PSPLFFTScanner``); this is a numpy-only reimplementation in aethra's idiom,
kept here rather than imported so aethra does not acquire that package's jax and
VBMicrolensing dependencies for a seeding routine. Measured against the u0
multi-start on 28 events, seeding this way was 4x faster end to end and never
reached a worse chi2; on events whose season-scoped fit had failed the chi2 gate
it reached a better one.

The seed is deliberately coarse. It is meant to start a fit on the original
timestamps, not to replace one.
"""

import numpy as np

__all__ = [
    "pspl_excess_magnification",
    "pspl_fft_seed",
    "DEFAULT_U0_GRID",
    "DEFAULT_TEFF_GRID",
]

DEFAULT_U0_GRID = (0.01, 0.03, 0.1, 0.3, 0.6, 1.0)
DEFAULT_TEFF_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)


def pspl_excess_magnification(lag, u0, teff):
    """PSPL excess magnification ``A - 1`` as a function of ``t - t0``.

    Computed from an algebraically rationalised form rather than as ``A``
    minus one. In the wings A approaches 1, and the direct subtraction there
    cancels away most of the significant digits of the very quantity the
    matched filter is trying to weigh.

    Parameters
    ----------
    lag : array_like
        Time offset ``t - t0``, in days.
    u0 : float
        Impact parameter, positive.
    teff : float
        Effective timescale ``u0 * tE``, in days, positive. This is the width
        the bump actually shows, which is what a template needs to match.

    Returns
    -------
    ndarray
        ``A - 1`` at each lag. Non-finite results are returned as 0.
    """
    u0 = float(u0)
    teff = float(teff)
    if not np.isfinite(u0) or u0 <= 0:
        raise ValueError("u0 must be positive and finite")
    if not np.isfinite(teff) or teff <= 0:
        raise ValueError("teff must be positive and finite")

    lag = np.asarray(lag, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        u = u0 * np.hypot(1.0, lag / teff)
        root = np.hypot(u, 2.0)
        excess = 4.0 / (u * root * (u * u + 2.0 + u * root))
    return np.nan_to_num(excess, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare(time, flux, flux_err, grid_dt, max_grid_points):
    """Bin the observations into the weighted sufficient statistics the scan needs.

    Irregular sampling is absorbed here: every template afterwards sees the
    same two binned arrays, so the per-template cost is independent of how many
    observations there were.
    """
    span = float(np.max(time) - np.min(time))
    n_grid = max(int(np.ceil(span / grid_dt)) + 1, 2)
    if n_grid > max_grid_points:
        return None

    t_start = float(np.min(time))
    bin_index = np.floor((time - t_start) / grid_dt + 0.5).astype(np.int64)
    bin_index = np.clip(bin_index, 0, n_grid - 1)

    weights = 1.0 / flux_err**2
    total_weight = float(np.sum(weights))
    if not np.isfinite(total_weight) or total_weight <= 0:
        return None

    mean_flux = float(np.sum(weights * flux) / total_weight)
    centered = flux - mean_flux
    null_chi2 = float(np.sum(weights * centered * centered))

    binned_weight = np.bincount(bin_index, weights=weights, minlength=n_grid)
    binned_flux = np.bincount(bin_index, weights=weights * centered, minlength=n_grid)

    n_fft = 1
    while n_fft < 3 * n_grid - 2:
        n_fft *= 2

    return {
        "t0_grid": t_start + grid_dt * np.arange(n_grid, dtype=float),
        "lags": (np.arange(2 * n_grid - 1, dtype=float) - (n_grid - 1)) * grid_dt,
        "total_weight": total_weight,
        "null_chi2": null_chi2,
        "fft_weight": np.fft.rfft(binned_weight, n=n_fft),
        "fft_flux": np.fft.rfft(binned_flux, n=n_fft),
        "n_fft": n_fft,
        "crop": slice(n_grid - 1, 2 * n_grid - 1),
    }


def _scan_template(prepared, u0, teff, positive_source, singular_rtol):
    """Profiled delta chi2 at every t0 on the grid, for one template shape."""
    template = pspl_excess_magnification(prepared["lags"], u0, teff)
    n_fft, crop = prepared["n_fft"], prepared["crop"]

    fft_template = np.fft.rfft(template[::-1], n=n_fft)
    fft_template2 = np.fft.rfft(np.square(template)[::-1], n=n_fft)

    # Correlating with the reversed template turns each of these into the
    # weighted sum over observations, evaluated at every t0 at once.
    qx = np.fft.irfft(prepared["fft_weight"] * fft_template, n=n_fft)[crop]
    qxx = np.fft.irfft(prepared["fft_weight"] * fft_template2, n=n_fft)[crop]
    sxy = np.fft.irfft(prepared["fft_flux"] * fft_template, n=n_fft)[crop]

    # Normal equations with the constant baseline already profiled out.
    sxx = qxx - np.square(qx) / prepared["total_weight"]
    scale = max(1.0, float(np.max(np.abs(qxx))))
    valid = np.isfinite(sxx) & np.isfinite(sxy) & (sxx > singular_rtol * scale)

    numerator = np.maximum(sxy, 0.0) if positive_source else sxy
    delta_chi2 = np.zeros(len(sxx), dtype=float)
    delta_chi2[valid] = np.square(numerator[valid]) / sxx[valid]
    return np.clip(delta_chi2, 0.0, max(prepared["null_chi2"], 0.0))


def pspl_fft_seed(time, flux, flux_err, u0_grid=DEFAULT_U0_GRID,
                  teff_grid=DEFAULT_TEFF_GRID, samples_per_teff=8.0,
                  positive_source=True, max_grid_points=1000000,
                  singular_rtol=1e-12, top_k=1):
    """Rank PSPL starting values by matched filter over a ``(u0, teff)`` bank.

    Parameters
    ----------
    time, flux, flux_err : array_like
        Observations in flux. Must be finite with positive errors; callers
        working in magnitudes convert first.
    u0_grid, teff_grid : sequence of float
        Template bank. ``teff_grid`` should span the bump widths worth finding;
        its smallest entry sets the calculation grid spacing and therefore the
        cost.
    samples_per_teff : float
        Grid samples across the narrowest template.
    positive_source : bool
        Project negative source-flux solutions to zero, so the filter responds
        to brightenings only.
    max_grid_points : int
        Refuse to allocate a grid larger than this; returns no seed instead.
    singular_rtol : float
        Relative floor below which a template is treated as degenerate with
        the constant baseline.
    top_k : int
        How many ranked seeds to return. One is usually enough — a second
        local fit costs as much as the entire scan and, measured, found
        nothing better.

    Returns
    -------
    dict
        ``seeds``
            ``(top_k, 3)`` array of ``(t0, tE, u0)`` rows, best first. Empty
            if no template produced a positive delta chi2.
        ``t0``, ``tE``, ``u0``, ``teff``
            The best seed, or NaN.
        ``delta_chi2``
            Improvement over a constant baseline at the best seed. This is a
            by-product of seeding, not a detection statistic — it assumes the
            PSPL shape, which is exactly the assumption detection must avoid.
        ``null_chi2``
            Chi2 of the constant baseline, for scale.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    ok = (np.isfinite(time) & np.isfinite(flux)
          & np.isfinite(flux_err) & (flux_err > 0))
    time, flux, flux_err = time[ok], flux[ok], flux_err[ok]

    empty = {"seeds": np.empty((0, 3)), "t0": np.nan, "tE": np.nan,
             "u0": np.nan, "teff": np.nan, "delta_chi2": np.nan,
             "null_chi2": np.nan}
    if len(time) < 2 or np.ptp(time) <= 0:
        return empty

    u0_values = np.asarray(u0_grid, dtype=float)
    teff_values = np.asarray(teff_grid, dtype=float)
    u0_values = u0_values[np.isfinite(u0_values) & (u0_values > 0)]
    teff_values = teff_values[np.isfinite(teff_values) & (teff_values > 0)]
    if len(u0_values) == 0 or len(teff_values) == 0:
        return empty

    grid_dt = float(np.min(teff_values)) / float(samples_per_teff)
    prepared = _prepare(time, flux, flux_err, grid_dt, int(max_grid_points))
    if prepared is None:
        return empty

    found = []
    for u0 in u0_values:
        for teff in teff_values:
            delta_chi2 = _scan_template(prepared, u0, teff,
                                        positive_source, singular_rtol)
            index = int(np.argmax(delta_chi2))
            if delta_chi2[index] > 0:
                found.append((float(delta_chi2[index]),
                              float(prepared["t0_grid"][index]),
                              float(teff / u0), float(u0), float(teff)))

    if not found:
        return {**empty, "null_chi2": prepared["null_chi2"]}

    found.sort(key=lambda row: row[0], reverse=True)
    found = found[:max(1, int(top_k))]
    seeds = np.array([[row[1], row[2], row[3]] for row in found], dtype=float)

    return {
        "seeds": seeds,
        "t0": found[0][1], "tE": found[0][2],
        "u0": found[0][3], "teff": found[0][4],
        "delta_chi2": found[0][0],
        "null_chi2": prepared["null_chi2"],
    }
