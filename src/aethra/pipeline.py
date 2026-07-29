"""Main per-DataFrame pipeline driver.

The pipeline runs in five stages, and the split between them is the point.
Detection (Stage 2) never touches a model, because a planetary or binary
anomaly is by definition a departure from PSPL and gating on PSPL quality
throws away exactly the events worth finding. PSPL enters only afterwards, as
a classification axis: how well it fits, and what the residuals look like when
it does not.

    Stage 1  renormalize_errors      baseline scatter -> error scale factor
    Stage 2  coherent_peak_scan      model-free detection, whole curve
             periodicity_with_event_masked, achromatic test
    Stage 3  fit_pspl_full           PSPL over the whole curve, FFT-seeded
    Stage 4  residual_structure      is what PSPL missed localized at t0?
    Stage 5  label                   see :data:`aethra.schema.LABELS`

Seasons survive only as reporting context and as the unit the achromatic test
compares within. Nothing in detection or fitting is scoped to one any more:
that scoping is why no fitted tE in the old catalogue exceeded 100 d while
true ones reach 723 d.
"""

import numpy as np
import pandas as pd

from .achromatic import achromatic_test_from_event
from .coherence import coherent_peak_scan
from .pspl import fit_pspl_full, renormalize_errors, residual_structure
from .schema import EVENT_LABELS, OUTPUT_COLUMNS
from .seasons import split_into_seasons
from .variability import periodicity_with_event_masked

__all__ = ["run_pipeline_from_dataframe", "run_and_report"]


def _empty_result(obj_name):
    """A row for an object we could not say anything about."""
    row = {col: np.nan for col in OUTPUT_COLUMNS}
    row.update({
        "name": obj_name, "label": "no_event",
        "is_candidate": False, "event_candidate": False,
        "is_ffp_candidate": False, "is_variable_star": False,
        "pspl_like": False, "non_pspl_candidate": False,
        "non_pspl_unexplained": False, "fit_failed": False,
        "scan_candidate": False, "hc_periodic": False,
        "veto_periodic": False, "veto_recurrent": False, "veto_chromatic": False,
        "veto_baseline_variable": False,
        "residual_significant": False, "residual_localized": False,
        "fit_degenerate": False,
        "bump_flag": False, "is_flat": True, "n_up": 0, "n_nights": 0,
    })
    return row


def _baseline_and_peak_mag(time, mag, peak_time, duration_days):
    """Magnitudes outside and at the top of the detected excursion.

    Reported for continuity with the old season-scoped output, but measured
    over the whole curve: the baseline is the median away from the event, the
    peak the brightest point within it.
    """
    if len(mag) == 0:
        return np.nan, np.nan
    inside = np.zeros(len(time), dtype=bool)
    if np.isfinite(peak_time) and np.isfinite(duration_days) and duration_days > 0:
        inside = np.abs(time - peak_time) <= duration_days
    if not inside.any() or inside.all():
        return float(np.median(mag)), float(np.min(mag))
    return float(np.median(mag[~inside])), float(np.min(mag[inside]))


def _season_of(time, season_ids, peak_time):
    """Which season the detected peak fell in, for reporting."""
    if not np.isfinite(peak_time) or len(time) == 0:
        return np.nan
    return season_ids[int(np.argmin(np.abs(time - peak_time)))]


def _classify(scan_candidate, is_variable_star, fit, structure):
    """Stage 5. Turn the measured axes into one label.

    The order matters and encodes the separation the redesign is for: whether
    an event happened is settled before PSPL is consulted, and a PSPL fit that
    fails to describe the curve downgrades nothing — it routes the object to
    ``non_pspl_*`` instead of discarding it.
    """
    if is_variable_star:
        return "variable_star"
    if not scan_candidate:
        return "no_event"
    if fit is None:
        return "fit_failed"
    if structure["residual_significant"]:
        return "non_pspl_candidate" if structure["residual_localized"] else "non_pspl_unexplained"
    return "pspl_like"


def run_pipeline_from_dataframe(df, config, debug=False):
    """
    Run the microlensing pipeline on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the columns named in config['time_col'],
        config['mag_col'], and config['err_col'].
    config : dict
        See example_run.ipynb (Configuration cell) for all keys. Beyond the
        input-column names the ones that change what gets found are
        ``min_peak_score`` (detection threshold, default 40),
        ``residual_min_peak_score`` / ``residual_localization_tE`` (what counts
        as anomalous residual structure), ``max_error_renorm`` (above which the
        baseline is not a baseline), and ``max_tE_over_duration`` (above which
        the fitted timescale is flagged degenerate).
    debug : bool
        If True, print a per-object breakdown of each stage's verdict.

    Returns
    -------
    pd.DataFrame with :data:`aethra.schema.OUTPUT_COLUMNS`.
    """
    time_col   = config["time_col"]
    mag_col    = config["mag_col"]
    err_col    = config["err_col"]
    group_col  = config.get("group_col")
    filter_col = config.get("filter_col")
    target_filter    = config.get("target_filter",    "F146")
    primary_filter   = config.get("primary_filter",   "F146")
    _sf = config.get("secondary_filters", config.get("secondary_filter", None))
    secondary_filters = [_sf] if isinstance(_sf, str) else (_sf or [])
    min_points        = config.get("min_points",           10)
    gap_days          = config.get("season_gap_days",     100)
    ffp_tE_max        = config.get("ffp_tE_max",          2.0)
    chromatic_min_pts = config.get("chromatic_min_points",  5)
    min_peak_score    = config.get("min_peak_score",       40.0)
    max_error_renorm  = config.get("max_error_renorm",   1000.0)
    residual_min_peak = config.get("residual_min_peak_score", 40.0)
    localization_tE   = config.get("residual_localization_tE", 2.0)
    max_tE_over_dur   = config.get("max_tE_over_duration",    3.0)

    for col in [time_col, mag_col, err_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame. "
                             f"Available columns: {list(df.columns)}")

    working = df.copy()
    grouped = [("single_object", working)] if group_col is None else working.groupby(group_col)

    results = []

    for obj_name, obj_df_all in grouped:
        obj_df_all = obj_df_all.sort_values(time_col).copy()
        if len(obj_df_all) == 0:
            continue

        season_labels = split_into_seasons(obj_df_all[time_col].to_numpy(), gap_days=gap_days)
        obj_df_all["season_id"] = season_labels
        n_seasons = int(obj_df_all["season_id"].nunique())

        if filter_col is not None and filter_col in obj_df_all.columns:
            obj_df_primary = obj_df_all[obj_df_all[filter_col] == target_filter].copy()
        else:
            obj_df_primary = obj_df_all.copy()

        if len(obj_df_primary) < min_points:
            if debug:
                print(f"[{obj_name}] SKIP — {len(obj_df_primary)} points in "
                      f"primary band '{target_filter}', need {min_points}")
            results.append({**_empty_result(obj_name), "n_seasons": n_seasons})
            continue

        time = obj_df_primary[time_col].to_numpy(dtype=float)
        mag = obj_df_primary[mag_col].to_numpy(dtype=float)
        mag_err = obj_df_primary[err_col].to_numpy(dtype=float)

        # --- Stage 2: detection, no model, no season scoping ------------------
        scan = coherent_peak_scan(time, mag, mag_err)
        scan_candidate = bool(np.isfinite(scan["peak_score"])
                              and scan["peak_score"] > min_peak_score)

        best_season = _season_of(time, obj_df_primary["season_id"].to_numpy(),
                                 scan["peak_time"])
        is_achromatic = achromatic_test_from_event(
            obj_df=obj_df_all,
            time_col=time_col, mag_col=mag_col, err_col=err_col,
            filter_col=filter_col,
            season_col="season_id",
            best_season=best_season,
            t0_guess_raw=scan["peak_time"],
            best_window_days=scan["main_duration_days"],
            primary_filter=primary_filter,
            secondary_filters=secondary_filters,
            min_points=chromatic_min_pts,
        )
        veto_chromatic = (is_achromatic is False)

        # Recurrence is reported, not vetoed on. Measured on 195 detected
        # events and 3 detected LPVs, requiring n_up == 1 would have removed
        # 8 of the 23 events with tE > 50 d to remove those 3 LPVs: a long
        # event straddling a seasonal gap fragments into several excursions
        # for the same reason a variable does.
        veto_recurrent = bool(scan["n_up"] > 1)

        # --- Stage 1: what the errors are actually worth ---------------------
        renorm = renormalize_errors(time, mag, mag_err,
                                    peak_time=scan["peak_time"],
                                    duration_days=scan["main_duration_days"])

        # The factor is also the sharpest variable-star statistic there is, and
        # it says something the periodicity test cannot: the curve away from the
        # event is not constant either. That is the definition of a variable and
        # it needs no period, which is why it catches the semi-regular LPVs that
        # fold badly. Measured on 204 events and 310 variables the two
        # populations are separated by two orders of magnitude on each side of
        # this default — no detected event exceeded 645, no detected variable
        # came in under 92000 — but that is five variables' worth of evidence,
        # so it is a config key.
        veto_baseline_variable = bool(renorm["error_renorm"] > max_error_renorm)

        # --- Stage 3 + 4: PSPL as a classification axis ----------------------
        # Objects already settled as variables by their own baseline are not
        # fitted: the fit is the expensive step, they are the slowest cases,
        # and what it returns for them is meaningless anyway (a chi2_red of
        # 2.5e5 at a tE read off whichever hump the filter locked onto).
        fit, structure = None, None
        if scan_candidate and not veto_baseline_variable:
            fit = fit_pspl_full(time, mag, mag_err,
                                t0_guess=scan["peak_time"],
                                duration_days=scan["main_duration_days"],
                                error_renorm=renorm["error_renorm"])
            if fit is not None:
                structure = residual_structure(
                    fit["residual_time"], fit["residual_flux"],
                    t0_fit=fit["t0_fit"], tE_fit=fit["tE_fit"],
                    localization_tE=localization_tE,
                    min_peak_score=residual_min_peak,
                )

        # tE and u0 trade off along a nearly constant teff = u0 * tE, and the
        # low-u0 branch of that valley can win on chi2 while describing the
        # curve no better: the worst case measured put tE_fit = 3425 d on a
        # 64.8 d event at u0_fit = 1e-4, the fit's own lower bound, with a
        # perfectly ordinary chi2_red of 1.05. Stage 2 already measured how wide
        # the excursion is without any model, and a PSPL bump is never narrower
        # than its own tE, so a fit claiming tE longer than the excursion it is
        # supposed to describe is self-inconsistent. Measured over the 2218
        # fitted events of the full run, in which 49 fits are truly wrong by
        # more than 3x and 15 by more than 10x, the threshold trades reach
        # against purity:
        #
        #     ratio > 1x   flags  49 (2.2%)   purity 0.449   catches 15/15
        #     ratio > 2x   flags  19 (0.9%)   purity 0.737   catches 12/15
        #     ratio > 3x   flags  13 (0.6%)   purity 0.923   catches 10/15
        #     ratio > 5x   flags  10 (0.5%)   purity 0.900   catches  8/15
        #
        # The default is 3x. At 1x the flag fires on more well-measured events
        # than bad ones, which would teach a reader to ignore it. It is
        # reported, not vetoed: the event is real and detected, it is the
        # timescale the data do not pin down.
        fit_degenerate = bool(
            fit is not None and np.isfinite(fit["tE_fit"])
            and np.isfinite(scan["main_duration_days"]) and scan["main_duration_days"] > 0
            and fit["tE_fit"] > max_tE_over_dur * scan["main_duration_days"]
        )

        # Periodicity comes after the fit because the mask needs the event's
        # real extent, and the excursion width underestimates it badly for long
        # events: the robust baseline is measured from the event's own wings, so
        # a tE = 700 d event reports a ~180 d excursion. Masking only that left
        # enough of the rising and falling wings behind to look like a ~190 d
        # period, and 5 of 23 real events with tE > 50 d were being labelled
        # variable stars because of it.
        mask_days = scan["main_duration_days"]
        mask_center = scan["peak_time"]
        if fit is not None and np.isfinite(fit["tE_fit"]):
            mask_days = np.nanmax([mask_days, 2.0 * fit["tE_fit"]])
            mask_center = fit["t0_fit"]
        periodicity = periodicity_with_event_masked(
            time, mag, mag_err,
            peak_time=mask_center, duration_days=mask_days,
        )
        hc_periodic = bool(periodicity["high_confidence_periodic"])

        # A confidently periodic object, or one whose baseline is not a
        # baseline, is a variable star whether or not the coherence scan called
        # it an event; a chromatic veto only means anything about an event that
        # was detected.
        is_variable_star = bool(hc_periodic or veto_baseline_variable
                                or (scan_candidate and veto_chromatic))

        # --- Stage 5: labelling ----------------------------------------------
        label = _classify(scan_candidate, is_variable_star, fit, structure)
        event_candidate = label in EVENT_LABELS
        is_ffp_candidate = bool(
            event_candidate and fit is not None
            and np.isfinite(fit["tE_fit"]) and fit["tE_fit"] < ffp_tE_max
        )

        baseline_mag, peak_mag = _baseline_and_peak_mag(
            time, mag, scan["peak_time"], scan["main_duration_days"])

        if debug:
            print(
                f"[{obj_name}] label={label} | n_seasons={n_seasons} | "
                f"peak_score={scan['peak_score']:.1f} (>{min_peak_score} -> "
                f"{scan_candidate}), n_up={scan['n_up']} | "
                f"hc_periodic={hc_periodic}(P={periodicity['period']}), "
                f"chromatic={veto_chromatic}(is_achromatic={is_achromatic}) | "
                f"k={renorm['error_renorm']:.2f} | "
                + (f"tE={fit['tE_fit']:.2f} u0={fit['u0_fit']:.3f} "
                   f"chi2r={fit['chi2_red_pspl']:.2f} "
                   f"frac_expl={fit['frac_explained']:.3f}" if fit else "no fit")
                + (f" | resid score={structure['residual_peak_score']:.1f} "
                   f"offset={structure['residual_offset_tE']:.2f}tE"
                   if structure else "")
            )

        row = {
            "name": obj_name,
            "label": label,
            "is_candidate": event_candidate,
            "event_candidate": event_candidate,
            "is_ffp_candidate": is_ffp_candidate,
            "is_variable_star": is_variable_star,
            "pspl_like": label == "pspl_like",
            "non_pspl_candidate": label == "non_pspl_candidate",
            "non_pspl_unexplained": label == "non_pspl_unexplained",
            "fit_failed": label == "fit_failed",
            "scan_candidate": scan_candidate,
            "peak_score": scan["peak_score"],
            "peak_sigma": scan["peak_sigma"],
            "peak_time": scan["peak_time"],
            "onset_time": scan["onset_time"],
            "n_up": scan["n_up"],
            "pos_ratio": scan["pos_ratio"],
            "duty_cycle": scan["duty_cycle"],
            "main_duration_days": scan["main_duration_days"],
            "duration_fraction": scan["duration_fraction"],
            "n_nights": scan["n_nights"],
            "hc_periodic": hc_periodic,
            "period": periodicity["period"],
            "veto_periodic": hc_periodic,
            "veto_recurrent": veto_recurrent,
            "veto_chromatic": veto_chromatic,
            "veto_baseline_variable": veto_baseline_variable,
            "fit_degenerate": fit_degenerate,
            "is_achromatic": is_achromatic,
            "error_renorm": renorm["error_renorm"],
            "baseline_chi2_red": renorm["baseline_chi2_red"],
            "n_seasons": n_seasons,
            "best_season": best_season,
            "bump_flag": scan_candidate,
            "bump_snr": scan["peak_sigma"],
            "baseline_mag": baseline_mag,
            "peak_mag": peak_mag,
        }
        if fit is not None:
            row.update({
                "t0_fit": fit["t0_fit"] - 2450000 if np.isfinite(fit["t0_fit"]) else np.nan,
                "u0_fit": fit["u0_fit"],
                "tE_fit": fit["tE_fit"],
                "chi2_red_pspl": fit["chi2_red_pspl"],
                "chi2_red_pspl_renorm": fit["chi2_red_pspl_renorm"],
                "frac_explained": fit["frac_explained"],
                "chi2_flat": fit["chi2_flat"],
                "dof_flat": fit["dof_flat"],
                "chi2_red_flat": (fit["chi2_flat"] / fit["dof_flat"]
                                  if fit["dof_flat"] > 0 else np.nan),
                "is_flat": False,
            })
        else:
            row["is_flat"] = not scan_candidate
        if structure is not None:
            row.update({k: structure[k] for k in (
                "residual_peak_score", "residual_offset_tE",
                "residual_significant", "residual_localized")})
        else:
            row.update({"residual_significant": False, "residual_localized": False})

        results.append(row)

    result_df = pd.DataFrame(results)
    for col in OUTPUT_COLUMNS:
        if col not in result_df.columns:
            result_df[col] = np.nan
    return result_df[OUTPUT_COLUMNS]


def run_and_report(input_path, config, debug=False, output_csv=None):
    from aethra import load_and_run
    results = load_and_run(input_path, config, debug=debug)

    print(f"\nPipeline complete — {len(results)} object(s) processed.")
    print("  labels:")
    for label, count in results["label"].value_counts().items():
        print(f"    {label:<24} {count}")
    print(f"  event_candidate:  {results['event_candidate'].sum()}"
          f"   (is_candidate is the same column)")
    print(f"  is_ffp_candidate: {results['is_ffp_candidate'].sum()}")
    print(f"  vetoes: periodic={results['veto_periodic'].sum()} "
          f"chromatic={results['veto_chromatic'].sum()} "
          f"(recurrent={results['veto_recurrent'].sum()}, reported only)")

    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"  Saved → {output_csv}")

    return results
