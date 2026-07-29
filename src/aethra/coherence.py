"""Season-free coherent-excursion detection.

The routines here answer "is there a single, coherent, one-sided brightening
somewhere in this light curve?" without fitting any physical model and without
splitting the curve into seasons. That matters for two reasons:

* a long event (``tE`` of hundreds of days) spans several observing seasons, so
  any per-season scan sees it as several separate bumps;
* a bad PSPL fit is not evidence against microlensing, so detection must not
  consult PSPL at all.

The approach follows ``microeden``'s ``estimate_microlensing_peak``: bin to
observed nights, normalise robustly, smooth at several widths with a running
median, and score coherence as ``max(smoothed, 0) * sqrt(nights in window)``.
No regular grid is created and unobserved dates are never interpolated.
"""

import numpy as np

__all__ = [
    "daily_median_bins",
    "smooth_multiscale",
    "coherent_excursions",
    "coherent_peak_scan",
    "coherent_value_scan",
]

DEFAULT_HALF_WIDTHS = (2.0, 4.0, 7.0, 14.0, 30.0)


def daily_median_bins(time, values, bin_days=1.0):
    """Collapse observations onto observed nights by median.

    Only nights that actually contain observations produce a bin, so the
    output is still irregularly sampled.

    Parameters
    ----------
    time, values : array_like
        Observation times and values, sorted by time.
    bin_days : float
        Bin width in days.

    Returns
    -------
    tuple of ndarray
        ``(binned_time, binned_value, counts)``.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(time) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, np.array([], dtype=int)

    bin_id = np.floor(time / bin_days).astype(np.int64)
    _, starts, counts = np.unique(bin_id, return_index=True, return_counts=True)

    binned_time = np.empty(len(starts), dtype=float)
    binned_value = np.empty(len(starts), dtype=float)
    for i, (start, count) in enumerate(zip(starts, counts)):
        sl = slice(int(start), int(start) + int(count))
        binned_time[i] = np.median(time[sl])
        binned_value[i] = np.median(values[sl])

    return binned_time, binned_value, counts.astype(int)


def _running_median(binned_time, normalized, half_width, min_nights):
    """Running median and window occupancy over observed nights only."""
    lo = np.searchsorted(binned_time, binned_time - half_width, side="left")
    hi = np.searchsorted(binned_time, binned_time + half_width, side="right")
    counts = (hi - lo).astype(int)

    smoothed = np.full(len(binned_time), np.nan, dtype=float)
    enough = counts >= min_nights
    for i in np.flatnonzero(enough):
        smoothed[i] = np.median(normalized[lo[i]:hi[i]])
    return smoothed, counts


def smooth_multiscale(binned_time, normalized, half_widths=DEFAULT_HALF_WIDTHS,
                      min_nights=3):
    """Smooth at several widths and keep the width with the strongest peak.

    The coherence score ``max(smoothed, 0) * sqrt(counts)`` rewards excursions
    that are both significant and supported by many observed nights, so a
    single deviant point can never win.

    Returns
    -------
    dict
        Keys ``peak_score``, ``smoothed``, ``counts``, ``half_width``,
        ``peak_index``. ``smoothed`` is ``None`` when no width had enough
        nights anywhere.
    """
    best = {"peak_score": -np.inf, "smoothed": None, "counts": None,
            "half_width": np.nan, "peak_index": -1}

    for half_width in half_widths:
        smoothed, counts = _running_median(binned_time, normalized, half_width, min_nights)
        if not np.any(np.isfinite(smoothed)):
            continue
        score = np.maximum(np.nan_to_num(smoothed, nan=-np.inf), 0.0) * np.sqrt(counts)
        index = int(np.nanargmax(score))
        if float(score[index]) > best["peak_score"]:
            best = {"peak_score": float(score[index]), "smoothed": smoothed,
                    "counts": counts, "half_width": float(half_width),
                    "peak_index": index}

    if best["smoothed"] is None:
        best["peak_score"] = np.nan
    return best


def coherent_excursions(binned_time, smoothed, sigma=3.0, max_join_days=20.0):
    """Group nights above ``sigma`` into distinct excursions.

    Nights closer together than ``max_join_days`` belong to the same excursion,
    so an event interrupted by a seasonal gap is not split in two.

    Returns
    -------
    list of tuple
        ``(start_index, stop_index)`` pairs, inclusive, into ``binned_time``.
    """
    smoothed = np.asarray(smoothed, dtype=float)
    indices = np.flatnonzero(np.isfinite(smoothed) & (smoothed >= sigma))
    if len(indices) == 0:
        return []

    segments = []
    start = previous = indices[0]
    for index in indices[1:]:
        if binned_time[index] - binned_time[previous] > max_join_days:
            segments.append((int(start), int(previous)))
            start = index
        previous = index
    segments.append((int(start), int(previous)))
    return segments


def _center_and_scale(values):
    """Robust baseline and noise scale for a curve that may contain an event.

    Deliberately the plain MAD. A long event covering most of the window
    inflates the MAD with itself, and since magnification is one-sided the
    points at or below the centre are uncontaminated, so estimating the scale
    from that half alone looks like the better choice. Measured on the
    roman_simu sample (221 events, 81 with tE > 50 d, against 308 variables) it
    is not: the one-sided scale left recall unchanged at 0.928 while raising
    the variable false-positive rate from 0.010 to 0.016 and lowering the share
    of long events seen as a single excursion from 0.605 to 0.506. It sharpens
    the noise floor for variables as readily as for events.
    """
    center = float(np.median(values))

    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return center, scale


def _baseline_outside_peak(binned_time, binned_flux, best, sigma, max_join_days,
                           min_nights):
    """Re-estimate the baseline from the nights the main excursion does not cover.

    Returns ``None`` when the peak sits in no excursion, or when masking it
    would leave too few nights to measure anything from.
    """
    up = coherent_excursions(binned_time, best["smoothed"], sigma, max_join_days)
    peak_index = int(best["peak_index"])
    main = next(((a, b) for a, b in up if a <= peak_index <= b), None)
    if main is None:
        return None

    outside = np.ones(len(binned_time), dtype=bool)
    outside[main[0]:main[1] + 1] = False
    if outside.sum() < max(min_nights, 5):
        return None
    return _center_and_scale(binned_flux[outside])


def _empty_scan():
    return {
        "peak_score": np.nan, "peak_time": np.nan, "onset_time": np.nan,
        "peak_sigma": np.nan, "smoothing_half_width": np.nan,
        "strong_nights": 0, "n_up": 0, "n_down": 0,
        "pos_ratio": np.nan, "duty_cycle": np.nan,
        "main_duration_days": np.nan, "duration_fraction": np.nan,
        "observed_span_days": np.nan, "n_nights": 0,
    }


def coherent_peak_scan(time, mag, mag_err=None, bin_days=1.0,
                       half_widths=DEFAULT_HALF_WIDTHS, min_nights=3,
                       sigma=3.0, max_join_days=20.0, n_refine=0):
    """Scan a whole light curve for one coherent, one-sided brightening.

    Works in flux, because microlensing magnifies flux and the coherence score
    is meaningful only on a linear scale.

    Parameters
    ----------
    time, mag : array_like
        Observation times (days) and magnitudes.
    mag_err : array_like, optional
        Unused for the scan itself; accepted so callers can pass the usual
        triple. Non-finite errors do not remove a point here.
    bin_days : float
        Night-binning width.
    half_widths : sequence of float
        Smoothing half-widths in days.
    min_nights : int
        Minimum observed nights inside a smoothing window.
    sigma : float
        Threshold, in robust sigma, for a night to count as "in excursion".
        Applied to ``smoothed`` as given, so callers detect dimmings by
        passing the negated curve.
    max_join_days : float
        Excursions separated by less than this are merged. The default sits
        below a seasonal gap, so an event bright enough to persist across one
        counts once per season. Raising it keeps such events whole but also
        merges the genuinely recurrent bumps of a variable star, which is what
        the count exists to expose.
    n_refine : int
        Passes that recompute the baseline with the main excursion masked out,
        so a long event cannot set the baseline it is measured against.
        Defaults to none: on the roman_simu sample one pass left recall
        unchanged at 0.928 and tripled the variable false-positive rate
        (0.010 to 0.029), because masking a variable's brightest bump lets the
        remaining bumps stand out just as well.

    Returns
    -------
    dict
        ``peak_score``
            Coherence of the strongest positive excursion. This is the
            detection statistic; it never consults a PSPL fit.
        ``peak_time``, ``onset_time``
            Time of the peak and of the first night above the onset threshold.
        ``peak_sigma``
            Smoothed amplitude at the peak, in robust sigma.
        ``n_up``, ``n_down``
            Distinct positive / negative excursions. Microlensing gives
            ``n_up == 1``; oscillating variables give several or none.
        ``pos_ratio``
            Positive share of the smoothed area. Near 1 for a magnification,
            near 0.5 for a symmetric variable. Diagnostic only — thresholding
            on it costs recall without reducing false positives.
        ``duty_cycle``
            Fraction of observed nights inside a positive excursion.
        ``main_duration_days``, ``duration_fraction``
            Extent of the strongest excursion, absolute and relative to span.
    """
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)

    valid = np.isfinite(time) & np.isfinite(mag)
    if mag_err is not None:
        mag_err = np.asarray(mag_err, dtype=float)
        valid &= np.isfinite(mag_err) & (mag_err > 0)
    time, mag = time[valid], mag[valid]

    if len(time) < min_nights:
        return _empty_scan()

    return coherent_value_scan(
        time, 10 ** (-0.4 * mag), bin_days=bin_days, half_widths=half_widths,
        min_nights=min_nights, sigma=sigma, max_join_days=max_join_days,
        n_refine=n_refine,
    )


def coherent_value_scan(time, values, bin_days=1.0,
                        half_widths=DEFAULT_HALF_WIDTHS, min_nights=3,
                        sigma=3.0, max_join_days=20.0, n_refine=0):
    """``coherent_peak_scan`` on an already-linear signal.

    Separate from the magnitude entry point because PSPL residuals are the
    other thing worth scanning and they go negative, so they cannot be routed
    through a flux conversion. Everything downstream — binning, the robust
    baseline, the multi-scale smoothing — only ever assumes the input is
    linear, so the two share one implementation.

    Parameters and returns are those of :func:`coherent_peak_scan`, with
    ``values`` in place of ``mag``.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(time) & np.isfinite(values)
    time, values = time[valid], values[valid]
    if len(time) < min_nights:
        return _empty_scan()

    order = np.argsort(time, kind="stable")
    time, values = time[order], values[order]

    binned_time, binned_flux, _ = daily_median_bins(time, values, bin_days=bin_days)
    if len(binned_time) < min_nights:
        return _empty_scan()

    span = float(binned_time[-1] - binned_time[0])
    center, scale = _center_and_scale(binned_flux)
    n_refine = max(0, int(n_refine))

    # Every pass recomputes `normalized` and `best` together, so the two always
    # share one baseline; a new baseline only takes effect on the next pass.
    for refinement in range(n_refine + 1):
        normalized = (binned_flux - center) / scale
        best = smooth_multiscale(binned_time, normalized, half_widths, min_nights)
        if best["smoothed"] is None:
            return _empty_scan()
        if refinement == n_refine:
            break

        updated = _baseline_outside_peak(binned_time, binned_flux, best,
                                         sigma, max_join_days, min_nights)
        if updated is None or np.allclose(updated, (center, scale)):
            break
        center, scale = updated

    smoothed = best["smoothed"]
    finite = np.isfinite(smoothed)
    if not finite.any():
        return _empty_scan()

    up = coherent_excursions(binned_time, smoothed, sigma, max_join_days)
    down = coherent_excursions(binned_time, -smoothed, sigma, max_join_days)

    # Positive share of the smoothed area, integrated over observed nights.
    times_finite = binned_time[finite]
    values_finite = smoothed[finite]
    weights = np.gradient(times_finite) if len(times_finite) > 1 else np.ones(1)
    positive = float((np.maximum(values_finite, 0.0) * weights).sum())
    negative = float((np.maximum(-values_finite, 0.0) * weights).sum())
    total = positive + negative
    pos_ratio = positive / total if total > 0 else np.nan

    duty_cycle = float((values_finite >= sigma).mean())

    if up:
        strengths = [np.nanmax(smoothed[a:b + 1]) for a, b in up]
        a, b = up[int(np.argmax(strengths))]
        window = slice(a, b + 1)
        local_peak = a + int(np.nanargmax(smoothed[window]))
        peak_time = float(binned_time[local_peak])
        peak_sigma = float(smoothed[local_peak])
        main_duration = float(binned_time[b] - binned_time[a]) + 2.0 * best["half_width"]
        strong_nights = int(np.sum(normalized[window] >= sigma))
        onset_time = float(binned_time[a])
    else:
        peak_index = int(best["peak_index"])
        peak_time = float(binned_time[peak_index])
        peak_sigma = float(smoothed[peak_index]) if finite[peak_index] else np.nan
        main_duration = np.nan
        strong_nights = 0
        onset_time = np.nan

    return {
        "peak_score": float(best["peak_score"]),
        "peak_time": peak_time,
        "onset_time": onset_time,
        "peak_sigma": peak_sigma,
        "smoothing_half_width": float(best["half_width"]),
        "strong_nights": strong_nights,
        "n_up": len(up),
        "n_down": len(down),
        "pos_ratio": pos_ratio,
        "duty_cycle": duty_cycle,
        "main_duration_days": main_duration,
        "duration_fraction": main_duration / span if span > 0 else np.nan,
        "observed_span_days": span,
        "n_nights": len(binned_time),
    }
