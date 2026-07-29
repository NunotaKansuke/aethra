"""Point-source point-lens (PSPL) magnification model and fitting.

The fit is a classification axis, not a detection test. Whether an event
happened is settled by :mod:`aethra.coherence` without reference to any model;
what a fit adds is *which* model explains it. A curve PSPL cannot describe is
as interesting as one it can — a planetary or binary anomaly is by definition a
departure from PSPL — so nothing here decides candidacy on goodness of fit.
"""

import numpy as np
from scipy.optimize import least_squares

from .coherence import coherent_value_scan
from .pspl_seed import DEFAULT_TEFF_GRID, DEFAULT_U0_GRID, pspl_fft_seed

__all__ = [
    "pspl_magnification",
    "solve_fs_fb",
    "fit_pspl",
    "fit_pspl_candidate",
    "renormalize_errors",
    "fit_pspl_full",
    "residual_structure",
]


def pspl_magnification(t, t0, u0, tE):
    tau = (t - t0) / tE
    u = np.sqrt(u0**2 + tau**2)
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))


def solve_fs_fb(A, flux, flux_err):
    w = 1.0 / flux_err**2
    S_AA = np.sum(w * A * A)
    S_A1 = np.sum(w * A)
    S_11 = np.sum(w)
    S_AF = np.sum(w * A * flux)
    S_1F = np.sum(w * flux)
    M = np.array([[S_AA, S_A1], [S_A1, S_11]])
    b = np.array([S_AF, S_1F])
    try:
        Fs, Fb = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        Fs, Fb = np.nan, np.nan
    return Fs, Fb


def fit_pspl(time, mag, mag_err):
    time = np.asarray(time, dtype=float)
    mag  = np.asarray(mag,  dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)

    valid = np.isfinite(time) & np.isfinite(mag) & np.isfinite(mag_err) & (mag_err > 0)
    time, mag, mag_err = time[valid], mag[valid], mag_err[valid]

    if len(time) < 10 or np.ptp(time) <= 0:
        return None

    flux     = 10 ** (-0.4 * mag)
    flux_err = 0.4 * np.log(10) * flux * mag_err

    good = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    time, flux, flux_err = time[good], flux[good], flux_err[good]

    if len(time) < 10:
        return None

    peak_idx      = np.argmax(flux)
    t0_guess      = time[peak_idx]
    baseline_flux = np.median(np.sort(flux)[:max(10, len(flux) // 5)])
    peak_flux     = np.max(flux)
    half_level    = baseline_flux + 0.5 * (peak_flux - baseline_flux)
    above_half    = time[flux >= half_level]
    tE_guess      = (
        max((above_half.max() - above_half.min()) / 2.0, 1.0)
        if len(above_half) >= 2
        else max(np.ptp(time) / 20.0, 1.0)
    )
    u0_guess = 0.3

    def residuals(p):
        t0, tE, u0 = p
        if tE <= 0 or u0 <= 0:
            return np.full_like(flux, 1e6)
        A = pspl_magnification(time, t0, u0, tE)
        Fs, Fb = solve_fs_fb(A, flux, flux_err)
        if not np.isfinite(Fs) or not np.isfinite(Fb):
            return np.full_like(flux, 1e6)
        return (flux - (Fs * A + Fb)) / flux_err

    t0_pad = max(5.0, min(20.0, 0.25 * np.ptp(time)))
    bounds  = (
        [t0_guess - t0_pad, 0.1,  1e-3],
        [t0_guess + t0_pad, max(100.0, np.ptp(time)), 2.0],
    )

    try:
        res = least_squares(
            residuals,
            x0=np.array([t0_guess, tE_guess, u0_guess]),
            bounds=bounds,
            max_nfev=50000,
        )
    except Exception:
        return None

    if not res.success:
        return None

    t0_fit, tE_fit, u0_fit = res.x
    A_fit          = pspl_magnification(time, t0_fit, u0_fit, tE_fit)
    Fs_fit, Fb_fit = solve_fs_fb(A_fit, flux, flux_err)

    if not np.isfinite(Fs_fit) or not np.isfinite(Fb_fit):
        return None

    model_flux = Fs_fit * A_fit + Fb_fit
    chi2       = np.sum(((flux - model_flux) / flux_err) ** 2)
    dof        = len(flux) - 3
    chi2_red   = chi2 / dof if dof > 0 else np.nan

    return {"t0_fit": t0_fit, "tE_fit": tE_fit, "u0_fit": u0_fit, "chi2_red_pspl": chi2_red}


def renormalize_errors(time, mag, mag_err, peak_time=np.nan, duration_days=np.nan,
                       mask_factor=2.0, min_points=20, floor=1.0):
    """Scale factor correcting under-estimated photometric errors.

    A constant fitted to the out-of-event baseline should give a reduced chi2
    of 1. When it gives ``k > 1`` the quoted errors are too small by sqrt(k),
    and every chi2 downstream is inflated by the same factor — which is enough
    on its own to push an otherwise ordinary event past a chi2 threshold.

    Parameters
    ----------
    time, mag, mag_err : array_like
        The full light curve.
    peak_time, duration_days : float
        Event window to exclude, from :func:`aethra.coherence.coherent_peak_scan`.
        Non-finite values mean the whole curve is used as baseline.
    mask_factor : float
        Half-width of the excluded window, in units of ``duration_days``.
    min_points : int
        Below this many surviving baseline points the estimate is not made.
    floor : float
        Lower clamp on the returned factor. Defaults to 1.0: an excess of
        scatter over the quoted errors is evidence they are too small, but a
        deficit is not evidence they are too large — binned or correlated
        photometry produces ``k < 1`` routinely, and dividing by it would
        manufacture bad fits out of conservative error bars. The unclamped
        value is returned as ``baseline_chi2_red`` either way.

    Returns
    -------
    dict
        ``error_renorm``
            The factor to divide a reduced chi2 by (or ``sqrt`` of, to scale
            errors). 1.0 when it could not be measured.
        ``baseline_chi2_red``
            The raw, unclamped reduced chi2 of the baseline.
        ``n_baseline``
            Points the estimate used.
    """
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)

    valid = (np.isfinite(time) & np.isfinite(mag)
             & np.isfinite(mag_err) & (mag_err > 0))
    time, mag, mag_err = time[valid], mag[valid], mag_err[valid]

    if np.isfinite(peak_time) and np.isfinite(duration_days) and duration_days > 0:
        outside = np.abs(time - peak_time) > mask_factor * duration_days
        if outside.sum() >= min_points:
            mag, mag_err = mag[outside], mag_err[outside]

    empty = {"error_renorm": 1.0, "baseline_chi2_red": np.nan, "n_baseline": len(mag)}
    if len(mag) < min_points:
        return empty

    weights = 1.0 / mag_err**2
    wsum = float(np.sum(weights))
    if not np.isfinite(wsum) or wsum <= 0:
        return empty

    center = float(np.sum(weights * mag) / wsum)
    chi2 = float(np.sum(weights * (mag - center) ** 2))
    dof = len(mag) - 1
    if dof <= 0 or not np.isfinite(chi2):
        return empty

    baseline_chi2_red = chi2 / dof
    factor = max(float(floor), baseline_chi2_red)
    if not np.isfinite(factor) or factor <= 0:
        factor = 1.0

    return {"error_renorm": factor, "baseline_chi2_red": baseline_chi2_red,
            "n_baseline": len(mag)}


def fit_pspl_full(time, mag, mag_err, t0_guess=np.nan, duration_days=np.nan,
                  error_renorm=1.0, u0_grid=DEFAULT_U0_GRID,
                  teff_grid=DEFAULT_TEFF_GRID, top_k=1):
    """Fit PSPL to a whole light curve, started from an FFT matched filter.

    :func:`fit_pspl` is season-scoped and starts from the brightest point. Both
    hurt long events: a season cuts off the wings that constrain tE, and the
    brightest single point is a noise excursion as often as a peak. It is the
    season scoping that does the real damage — given the same whole curve the
    two optimisers agree — and it is why no fitted tE in the catalogue exceeds
    100 d while true ones reach 723 d.

    Starting values come from :func:`aethra.pspl_seed.pspl_fft_seed`, which
    scans t0 globally rather than descending from a guess. Measured on 28
    events this was 4x faster than a u0 multi-start and never reached a worse
    chi2; on events whose season-scoped fit failed the chi2 gate it reached a
    better one.

    Parameters
    ----------
    time, mag, mag_err : array_like
        The full light curve, all seasons.
    t0_guess, duration_days : float
        Peak time and excursion width from the coherence scan. Refined as an
        additional start alongside the matched filter's seeds, and used alone
        if the filter finds nothing above a constant baseline.
    error_renorm : float
        Factor from :func:`renormalize_errors`. Divides the reported reduced
        chi2; it cannot change the best-fit parameters, since scaling all
        errors by a constant leaves the weighted residuals' minimum where it
        was.
    u0_grid, teff_grid : sequence of float
        Template bank for the seeding scan.
    top_k : int
        Seeds to refine. Each costs about as much as the whole scan, and a
        second one was measured to find nothing better.

    Returns
    -------
    dict or None
        ``None`` if no start converged. Otherwise the fitted parameters, the
        reduced chi2 both raw and renormalised, ``frac_explained``, and the
        flux-space residuals with their times, for residual analysis.
    """
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)

    valid = (np.isfinite(time) & np.isfinite(mag)
             & np.isfinite(mag_err) & (mag_err > 0))
    time, mag, mag_err = time[valid], mag[valid], mag_err[valid]

    if len(time) < 10 or np.ptp(time) <= 0:
        return None

    order = np.argsort(time, kind="stable")
    time, mag, mag_err = time[order], mag[order], mag_err[order]

    flux = 10 ** (-0.4 * mag)
    flux_err = 0.4 * np.log(10) * flux * mag_err

    good = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    time, flux, flux_err = time[good], flux[good], flux_err[good]
    if len(time) < 10:
        return None

    span = float(np.ptp(time))

    seed = pspl_fft_seed(time, flux, flux_err, u0_grid=u0_grid,
                         teff_grid=teff_grid, top_k=top_k)

    # The detection's own guess goes in alongside the matched filter's. The two
    # fail in different places, which is the point of keeping both: the filter
    # searches t0 globally but its bank is gridded in teff, so a bump narrower
    # than the narrowest template ranks a string of near-identical wrong-width
    # seeds above the right one, while the detection reads the width straight
    # off the excursion. Measured on 204 events that is one event in 204 - but
    # raising top_k enough to catch it costs 5x, and this costs one extra fit.
    detection_seed = []
    if np.isfinite(t0_guess) or len(seed["seeds"]) == 0:
        t0_start = t0_guess if np.isfinite(t0_guess) else float(time[np.argmax(flux)])
        if np.isfinite(duration_days) and duration_days > 0:
            tE_start = float(np.clip(duration_days / 2.0, 0.5, span))
        else:
            tE_start = max(span / 20.0, 1.0)
        detection_seed = [[t0_start, tE_start, 0.3]]

    seeds = np.array(list(seed["seeds"]) + detection_seed, dtype=float)
    if len(seeds) == 0:
        return None

    def residuals_for(p):
        t0, tE, u0 = p
        if tE <= 0 or u0 <= 0:
            return np.full_like(flux, 1e6)
        A = pspl_magnification(time, t0, u0, tE)
        Fs, Fb = solve_fs_fb(A, flux, flux_err)
        if not np.isfinite(Fs) or not np.isfinite(Fb):
            return np.full_like(flux, 1e6)
        return (flux - (Fs * A + Fb)) / flux_err

    best = None
    for t0_start, tE_start, u0_start in seeds:
        # The window is sized from the seed's own tE: a seed for a long event
        # needs room to move, a seed for a short one must not wander off it.
        pad = max(10.0, 2.0 * tE_start)
        bounds = (
            [t0_start - pad, 0.1, 1e-4],
            [t0_start + pad, max(100.0, 2.0 * span), 3.0],
        )
        start = np.clip(
            [t0_start, tE_start, u0_start],
            [bounds[0][0] + 1e-6, 0.1 + 1e-6, 1.1e-4],
            [bounds[1][0] - 1e-6, bounds[1][1] - 1e-6, 3.0 - 1e-3],
        )
        try:
            res = least_squares(residuals_for, x0=start, bounds=bounds, max_nfev=50000)
        except Exception:
            continue
        if not res.success:
            continue
        chi2 = float(np.sum(res.fun**2))
        if best is None or chi2 < best[0]:
            best = (chi2, res.x)

    if best is None:
        return None

    chi2, (t0_fit, tE_fit, u0_fit) = best
    A_fit = pspl_magnification(time, t0_fit, u0_fit, tE_fit)
    Fs_fit, Fb_fit = solve_fs_fb(A_fit, flux, flux_err)
    if not np.isfinite(Fs_fit) or not np.isfinite(Fb_fit):
        return None

    model_flux = Fs_fit * A_fit + Fb_fit
    dof = len(flux) - 3
    chi2_red = chi2 / dof if dof > 0 else np.nan

    weights = 1.0 / flux_err**2
    flat_level = float(np.sum(weights * flux) / np.sum(weights))
    chi2_flat = float(np.sum(weights * (flux - flat_level) ** 2))
    frac_explained = 1.0 - chi2 / chi2_flat if chi2_flat > 0 else np.nan

    renorm = float(error_renorm) if np.isfinite(error_renorm) and error_renorm > 0 else 1.0

    return {
        "t0_fit": t0_fit, "tE_fit": tE_fit, "u0_fit": u0_fit,
        "Fs_fit": float(Fs_fit), "Fb_fit": float(Fb_fit),
        "chi2_pspl": chi2, "dof_pspl": dof,
        "chi2_red_pspl": chi2_red,
        "chi2_red_pspl_renorm": chi2_red / renorm if np.isfinite(chi2_red) else np.nan,
        "chi2_flat": chi2_flat, "dof_flat": len(flux) - 1,
        "frac_explained": frac_explained,
        "n_points": len(flux),
        "residual_time": time,
        "residual_flux": flux - model_flux,
        "residual_sigma": (flux - model_flux) / flux_err,
    }


def residual_structure(residual_time, residual_flux, t0_fit=np.nan, tE_fit=np.nan,
                       localization_tE=2.0, min_peak_score=40.0, **scan_kwargs):
    """Whether what PSPL failed to explain is localised at the peak.

    This is what separates the two reasons a PSPL fit can be poor. A planetary
    or binary anomaly is a coherent excursion sitting within an einstein time
    or two of t0; systematics and an unmodelled variable star leave residual
    structure spread across the whole curve. Reduced chi2 alone cannot tell
    them apart, which is why gating on it discards real planets.

    Runs the coherence scan on the residuals in both signs, since an anomaly
    may brighten (a caustic crossing) or dim (a caustic-exit trough).

    Parameters
    ----------
    residual_time, residual_flux : array_like
        From :func:`fit_pspl_full`.
    t0_fit, tE_fit : float
        Fitted peak time and einstein time, defining what "at the peak" means.
    localization_tE : float
        How many einstein times from t0 still counts as localised.
    min_peak_score : float
        Coherence the residual excursion must reach before ``residual_localized``
        can be true. Required because proximity alone is not evidence: a clean
        PSPL fit leaves pure noise, whose strongest excursion lands within an
        einstein time of t0 about as often as not. The same threshold used for
        detection separates the two by roughly two orders of magnitude.
    **scan_kwargs
        Passed to :func:`aethra.coherence.coherent_value_scan`.

    Returns
    -------
    dict
        ``residual_peak_score``
            Coherence of the strongest residual excursion, either sign.
        ``residual_sign``
            ``+1`` if that excursion is a brightening, ``-1`` if a dimming.
        ``residual_offset_tE``
            Distance from t0 to the residual peak, in einstein times.
        ``residual_localized``
            Whether a *significant* residual sits within ``localization_tE``
            of t0 — the signature of a planetary or binary anomaly.
        ``residual_significant``
            Whether the residual excursion cleared ``min_peak_score`` at all.
            Significant but not localised means systematics or a variable.
        ``residual_n_up``
            Distinct positive residual excursions.
    """
    residual_flux = np.asarray(residual_flux, dtype=float)

    up = coherent_value_scan(residual_time, residual_flux, **scan_kwargs)
    down = coherent_value_scan(residual_time, -residual_flux, **scan_kwargs)

    up_score = up["peak_score"] if np.isfinite(up["peak_score"]) else -np.inf
    down_score = down["peak_score"] if np.isfinite(down["peak_score"]) else -np.inf
    scan, sign = (up, 1) if up_score >= down_score else (down, -1)

    offset = np.nan
    if np.isfinite(scan["peak_time"]) and np.isfinite(t0_fit) and tE_fit > 0:
        offset = abs(float(scan["peak_time"]) - float(t0_fit)) / float(tE_fit)

    peak_score = float(scan["peak_score"])
    significant = bool(np.isfinite(peak_score) and peak_score > min_peak_score)

    return {
        "residual_peak_score": peak_score,
        "residual_peak_time": float(scan["peak_time"]),
        "residual_sign": sign if np.isfinite(peak_score) else 0,
        "residual_offset_tE": offset,
        "residual_significant": significant,
        "residual_localized": bool(
            significant and np.isfinite(offset) and offset <= localization_tE
        ),
        "residual_n_up": int(up["n_up"]),
        "residual_duration_days": float(scan["main_duration_days"]),
    }


def fit_pspl_candidate(time, mags, mag_err, good_pspl_chi2=2.5, ffp_tE_max=2.0):
    """Season-scoped fit plus a chi2 gate. Kept for callers written against the
    old schema; the pipeline no longer uses it.

    The gate this applies is the thing the redesign removed. It answers "does
    PSPL describe this?" and then discards whatever says no, which is exactly
    backwards for anomalies: measured on 89 events it had rejected, 39 have
    residual structure localized at t0. Use :func:`fit_pspl_full` together with
    :func:`residual_structure` instead, and read the verdict off the pipeline's
    ``label`` column.
    """
    pspl_result = fit_pspl(time, mags, mag_err)
    if pspl_result is None:
        return {"t0_fit_raw": np.nan, "u0_fit": np.nan, "tE_fit": np.nan,
                "chi2_red_pspl": np.nan, "is_candidate": False, "is_ffp_candidate": False}

    good_pspl_fit = (
        np.isfinite(pspl_result["tE_fit"])
        and np.isfinite(pspl_result["chi2_red_pspl"])
        and pspl_result["chi2_red_pspl"] < good_pspl_chi2
    )
    is_candidate     = bool(good_pspl_fit)
    is_ffp_candidate = bool(is_candidate and pspl_result["tE_fit"] < ffp_tE_max)

    return {
        "t0_fit_raw":    pspl_result["t0_fit"],
        "u0_fit":        pspl_result["u0_fit"],
        "tE_fit":        pspl_result["tE_fit"],
        "chi2_red_pspl": pspl_result["chi2_red_pspl"],
        "is_candidate":     is_candidate,
        "is_ffp_candidate": is_ffp_candidate,
    }
