"""Variability and Lomb-Scargle periodicity tests."""

import warnings

import numpy as np
from astropy.timeseries import LombScargle

from .detection import has_consecutive_outliers

__all__ = [
    "is_non_flat_lightcurve",
    "lomb_scargle_test",
    "is_periodic",
    "periodic_veto_from_other_seasons",
    "analyze_periodicity",
    "periodicity_with_event_masked",
]

PERIODICITY_DEFAULTS = {
    "min_points": 20,
    "min_span_days": 10.0,
    "min_period": 0.2,
    "max_period": 400.0,
    "max_period_fraction": 0.5,
    "n_frequencies": 512,
    "phase_bins": 20,
    "max_fap": 1e-6,
    "min_power": 0.3,
    "min_cycles": 3.0,
    "min_phase_coverage": 0.5,
    "min_variance_reduction": 0.3,
    "max_alias_score": 0.9,
}


def is_non_flat_lightcurve(mag, sigma_thresh=3, use_robust=True, n_consecutive=3):
    mag = np.asarray(mag, dtype=float)
    if len(mag) == 0 or not np.all(np.isfinite(mag)):
        return False
    med = np.median(mag)
    if use_robust:
        mad = np.median(np.abs(mag - med))
        sigma = 1.4826 * mad
    else:
        sigma = np.std(mag)
    if not np.isfinite(sigma) or sigma <= 0:
        return False
    mask = np.abs(mag - med) > sigma_thresh * sigma
    return has_consecutive_outliers(mask, n_consecutive=n_consecutive)


def lomb_scargle_test(time, mag, mag_err=None, min_period=0.1, max_period=100, min_points=10):
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)

    if mag_err is not None:
        mag_err = np.asarray(mag_err, dtype=float)
        valid = np.isfinite(time) & np.isfinite(mag) & np.isfinite(mag_err) & (mag_err > 0)
    else:
        valid = np.isfinite(time) & np.isfinite(mag)

    time = time[valid]
    mag = mag[valid]
    if mag_err is not None:
        mag_err = mag_err[valid]

    if len(mag) < min_points or np.ptp(time) <= 0 or np.ptp(mag) <= 0:
        return np.nan, np.nan, np.nan

    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period

    try:
        ls = LombScargle(time, mag, mag_err)
        frequency, power = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
        )
    except Exception:
        return np.nan, np.nan, np.nan

    finite = np.isfinite(frequency) & np.isfinite(power)
    frequency = frequency[finite]
    power = power[finite]

    if len(power) == 0:
        return np.nan, np.nan, np.nan

    idx = np.argmax(power)
    max_power = power[idx]
    best_frequency = frequency[idx]

    if not np.isfinite(max_power) or not np.isfinite(best_frequency) or best_frequency <= 0:
        return np.nan, np.nan, np.nan

    best_period = 1.0 / best_frequency

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            fap = ls.false_alarm_probability(max_power)
    except Exception:
        fap = np.nan

    return best_period, fap, max_power


def is_periodic(time, mag, mag_err=None, fap_threshold=0.01, min_points=10,
                min_period=0.1, max_period=100):
    period, fap, max_power = lomb_scargle_test(
        time, mag, mag_err,
        min_period=min_period, max_period=max_period, min_points=min_points,
    )
    if not np.isfinite(period) or not np.isfinite(fap):
        return False, period, fap
    return (fap < fap_threshold), period, fap


def _one_day_alias_score(period):
    """1.0 when the period sits on a 0.5/1/2 day sampling alias, 0 when far away."""
    aliases = np.array([0.5, 1.0, 2.0], dtype=float)
    log_distance = float(np.min(np.abs(np.log(period / aliases))))
    return float(np.exp(-0.5 * (log_distance / 0.03) ** 2))


def _empty_periodicity(reason):
    return {
        "valid": False, "reason": reason, "period": np.nan, "power": 0.0,
        "fap": 1.0, "phase_coverage": 0.0, "cycles": 0.0,
        "variance_reduction": 0.0, "alias_score": 0.0,
        "high_confidence_periodic": False,
    }


def analyze_periodicity(time, values, errors=None, **overrides):
    """Full-curve periodicity test with corroborating evidence beyond the FAP.

    :func:`is_periodic` thresholds the Lomb-Scargle false-alarm probability
    alone, which is unreliable on real photometry: correlated noise and a
    single smooth bump both produce tiny FAPs. This test additionally requires
    the folded curve to look genuinely periodic — enough cycles observed,
    phase coverage that is not clumped, and a real reduction in scatter when
    folded — and rejects the 0.5/1/2 day sampling aliases.

    Unlike :func:`periodic_veto_from_other_seasons` this runs on the whole
    curve at once and searches out to ``max_period`` days (400 by default), so
    long-period variables are actually reachable. The per-season test caps the
    period at ``min(50, 0.7 * season length)`` and therefore cannot see them.

    Parameters
    ----------
    time, values : array_like
        Observation times in days and values on a linear scale (flux, not
        magnitudes, if the result is to be compared against a magnification).
    errors : array_like, optional
        Per-point uncertainties. Points with non-finite or non-positive errors
        are dropped when errors are supplied.
    **overrides
        Any key of :data:`PERIODICITY_DEFAULTS`.

    Returns
    -------
    dict
        ``high_confidence_periodic`` is the conservative AND of every
        criterion, intended as evidence for a variable-star label rather than
        as a veto on event detection.
    """
    config = dict(PERIODICITY_DEFAULTS)
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(f"Unknown periodicity option(s): {sorted(unknown)}")
    config.update(overrides)

    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(time) & np.isfinite(values)
    if errors is not None:
        errors = np.asarray(errors, dtype=float)
        finite &= np.isfinite(errors) & (errors > 0)
    time, values = time[finite], values[finite]
    errors = None if errors is None else errors[finite]

    if len(time) < config["min_points"]:
        return _empty_periodicity("too few observations")

    order = np.argsort(time, kind="stable")
    time, values = time[order], values[order]
    errors = None if errors is None else errors[order]

    unique = np.concatenate(([True], np.diff(time) > 0))
    time, values = time[unique], values[unique]
    errors = None if errors is None else errors[unique]

    span = float(time[-1] - time[0])
    if span < config["min_span_days"]:
        return _empty_periodicity("observed span too short")

    max_period = min(config["max_period"], span * config["max_period_fraction"])
    if max_period <= config["min_period"]:
        return _empty_periodicity("empty period range")

    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values)) or 1.0
    normalized = np.clip((values - center) / scale, -20.0, 20.0)

    frequency = np.geomspace(1.0 / max_period, 1.0 / config["min_period"],
                             int(config["n_frequencies"]))
    dy = None if errors is None else np.clip(errors / scale, 1e-4, 20.0)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            periodogram = LombScargle(time - time[0], normalized, dy=dy)
            power = np.asarray(periodogram.power(frequency), dtype=float)
    except Exception:
        return _empty_periodicity("periodogram failed")

    if not np.any(np.isfinite(power)):
        return _empty_periodicity("periodogram failed")

    best = int(np.nanargmax(power))
    max_power = float(np.clip(power[best], 0.0, 1.0))
    period = float(1.0 / frequency[best])

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            fap = float(np.clip(periodogram.false_alarm_probability(max_power), 0.0, 1.0))
    except Exception:
        fap = 1.0

    # Fold and measure how much scatter the periodic model actually removes.
    phase = np.mod(time - time[0], period) / period
    bins = int(config["phase_bins"])
    phase_index = np.minimum((phase * bins).astype(int), bins - 1)
    occupied = np.unique(phase_index)
    phase_coverage = float(len(occupied) / bins)

    folded = np.empty_like(normalized)
    for index in occupied:
        in_bin = phase_index == index
        folded[in_bin] = np.median(normalized[in_bin])
    total_variance = float(np.var(normalized))
    residual_variance = float(np.var(normalized - folded))
    variance_reduction = (
        float(np.clip(1.0 - residual_variance / total_variance, 0.0, 1.0))
        if total_variance > 0 else 0.0
    )

    cycles = span / period
    alias_score = _one_day_alias_score(period)

    high_confidence = bool(
        fap <= config["max_fap"]
        and max_power >= config["min_power"]
        and cycles >= config["min_cycles"]
        and phase_coverage >= config["min_phase_coverage"]
        and variance_reduction >= config["min_variance_reduction"]
        and alias_score < config["max_alias_score"]
    )

    return {
        "valid": True, "reason": "accepted", "period": period,
        "power": max_power, "fap": fap, "phase_coverage": phase_coverage,
        "cycles": cycles, "variance_reduction": variance_reduction,
        "alias_score": alias_score,
        "high_confidence_periodic": high_confidence,
    }


def periodicity_with_event_masked(time, values, errors=None, peak_time=np.nan,
                                  duration_days=np.nan, mask_factor=1.5,
                                  min_points=50, **overrides):
    """Run :func:`analyze_periodicity` with the candidate event removed.

    A single smooth brightening is itself a low-FAP signal, so testing the raw
    curve flags real events as periodic. Masking ``+-mask_factor * duration``
    around the peak first removes that confusion; on a mixed sample this drops
    the microlensing false-flag rate from roughly a quarter to a few percent
    while leaving long-period variables flagged.

    Falls back to the unmasked curve when the mask would leave too little data
    or when no event window is supplied.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    errors = None if errors is None else np.asarray(errors, dtype=float)

    keep = np.ones(len(time), dtype=bool)
    if np.isfinite(peak_time) and np.isfinite(duration_days) and duration_days > 0:
        candidate = np.abs(time - peak_time) > mask_factor * duration_days
        if candidate.sum() >= min_points:
            keep = candidate

    return analyze_periodicity(
        time[keep], values[keep],
        None if errors is None else errors[keep],
        **overrides,
    )


def periodic_veto_from_other_seasons(
    obj_df_primary, best_season,
    time_col="bjd", mag_col="mag", err_col="mag_err", season_col="season_id",
    min_points=40, fap_threshold=1e-6, min_period=0.1, max_period=50,
    min_cycles=4, ceiling_fraction=0.7,
):
    """
    Veto if any off-event season shows significant internal periodicity.
    Each season is tested independently so inter-season gaps never enter
    the Lomb-Scargle, avoiding survey-cadence aliases.
    """
    other_df = obj_df_primary[obj_df_primary[season_col] != best_season].copy()

    if len(other_df) < min_points:
        return False, np.nan, np.nan, "not enough off-event data"

    for season_id, season_df in other_df.groupby(season_col):
        if len(season_df) < min_points:
            continue

        time    = season_df[time_col].to_numpy(dtype=float)
        mag     = season_df[mag_col].to_numpy(dtype=float)
        mag_err = season_df[err_col].to_numpy(dtype=float)

        baseline = time.max() - time.min()
        season_max_period = min(max_period, ceiling_fraction * baseline)
        if season_max_period <= min_period:
            continue

        periodic_flag, period, fap = is_periodic(
            time, mag, mag_err,
            fap_threshold=fap_threshold,
            min_points=min_points,
            min_period=min_period,
            max_period=season_max_period,
        )

        if not periodic_flag:
            continue
        if not np.isfinite(period) or period <= 0:
            continue
        if period >= ceiling_fraction * season_max_period:
            continue

        n_cycles = baseline / period
        if n_cycles < min_cycles:
            continue

        return True, period, fap, f"repeating signal in off-event season {season_id}"

    return False, np.nan, np.nan, "no periodic off-event season found"
