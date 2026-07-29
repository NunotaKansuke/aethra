"""Output schema and label taxonomy shared across the pipeline."""

__all__ = ["OUTPUT_COLUMNS", "LABELS", "EVENT_LABELS"]

#: Every value ``label`` can take, in rough order of increasing interest.
#:
#: The pipeline used to answer one question — is this a candidate — with one
#: boolean, and a light curve that no model fit well was indistinguishable from
#: one with nothing in it. These separate the two axes that were being
#: conflated: whether there is an event at all (``no_event`` and
#: ``variable_star`` say there is not), and whether PSPL describes it.
LABELS = (
    "no_event",              # nothing coherent rose above the noise
    "variable_star",         # coherent, but periodic or chromatic
    "fit_failed",            # coherent, but no PSPL fit converged at all
    "pspl_like",             # coherent, PSPL describes it, residuals structureless
    "non_pspl_unexplained",  # coherent, residual structure is diffuse -> inspect
    "non_pspl_candidate",    # coherent, residual structure localized at t0 -> anomaly
)

#: Labels that assert an event happened. ``event_candidate`` is their union,
#: and ``is_candidate`` is kept as its alias for callers written against the
#: old schema.
EVENT_LABELS = ("pspl_like", "non_pspl_unexplained", "non_pspl_candidate")

OUTPUT_COLUMNS = [
    "name",
    # --- labels ---
    "label",
    "is_candidate",
    "event_candidate",
    "is_ffp_candidate",
    "is_variable_star",
    "pspl_like",
    "non_pspl_candidate",
    "non_pspl_unexplained",
    "fit_failed",
    # --- Stage 2: model-free detection ---
    "scan_candidate",
    "peak_score",
    "peak_sigma",
    "peak_time",
    "onset_time",
    "n_up",
    "pos_ratio",
    "duty_cycle",
    "main_duration_days",
    "duration_fraction",
    "n_nights",
    "hc_periodic",
    "period",
    # --- vetoes ---
    "veto_periodic",
    "veto_recurrent",
    "veto_chromatic",
    "veto_baseline_variable",
    "is_achromatic",
    # --- Stage 1: error renormalization ---
    "error_renorm",
    "baseline_chi2_red",
    # --- Stage 3: full-curve PSPL fit ---
    "t0_fit",
    "u0_fit",
    "tE_fit",
    "chi2_red_pspl",
    "chi2_red_pspl_renorm",
    "frac_explained",
    "fit_degenerate",
    # --- Stage 4: residual structure ---
    "residual_peak_score",
    "residual_offset_tE",
    "residual_significant",
    "residual_localized",
    # --- context ---
    "n_seasons",
    "best_season",
    "is_flat",
    "chi2_flat",
    "dof_flat",
    "chi2_red_flat",
    "bump_flag",
    "bump_snr",
    "baseline_mag",
    "peak_mag",
]
