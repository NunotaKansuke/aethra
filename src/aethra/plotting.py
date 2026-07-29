"""Matplotlib helpers for inspecting a light curve and its PSPL fit."""

from __future__ import annotations

import numpy as np

from .pspl import fit_pspl, pspl_magnification, solve_fs_fb

__all__ = ["plot_pspl_fit"]


def _model_magnitude(time, mag, mag_err, fit_result):
    """Return the PSPL model magnitude at ``time`` and at the data points."""
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)

    valid = np.isfinite(time) & np.isfinite(mag) & np.isfinite(mag_err) & (mag_err > 0)
    time, mag, mag_err = time[valid], mag[valid], mag_err[valid]
    flux = 10.0 ** (-0.4 * mag)
    flux_err = 0.4 * np.log(10.0) * flux * mag_err

    A_data = pspl_magnification(
        time, fit_result["t0_fit"], fit_result["u0_fit"], fit_result["tE_fit"]
    )
    Fs, Fb = solve_fs_fb(A_data, flux, flux_err)
    if not np.isfinite(Fs) or not np.isfinite(Fb):
        raise ValueError("Could not solve the PSPL source/blend fluxes for plotting")

    def model_at(model_time):
        A = pspl_magnification(
            model_time, fit_result["t0_fit"], fit_result["u0_fit"], fit_result["tE_fit"]
        )
        model_flux = Fs * A + Fb
        if np.any(model_flux <= 0) or not np.all(np.isfinite(model_flux)):
            raise ValueError("PSPL model produced a non-positive or non-finite flux")
        return -2.5 * np.log10(model_flux)

    return model_at, time, mag, mag_err


def plot_pspl_fit(
    time,
    mag,
    mag_err,
    *,
    fit_result=None,
    model_points=500,
    title="PSPL fit",
    save_path=None,
    ax=None,
):
    """Plot observed magnitudes, a PSPL fit, and the magnitude residuals.

    Parameters
    ----------
    time, mag, mag_err : array-like
        The light-curve data to fit and display.
    fit_result : dict, optional
        Result from :func:`aethra.pspl.fit_pspl`. If omitted, the fit is run
        on the supplied data.
    model_points : int, default=500
        Number of points used for the smooth model line.
    title : str, default="PSPL fit"
        Figure title.
    save_path : path-like, optional
        If supplied, save the figure to this path.
    ax : matplotlib.axes.Axes, optional
        Existing magnitude axis. A residual axis is created below it.

    Returns
    -------
    fig, axes, fit_result
        The Matplotlib figure, ``(magnitude_axis, residual_axis)``, and the
        fitted parameter dictionary.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plot_pspl_fit requires matplotlib; install it with `pip install matplotlib`"
        ) from exc

    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)
    valid = np.isfinite(time) & np.isfinite(mag) & np.isfinite(mag_err) & (mag_err > 0)
    if valid.sum() < 10:
        raise ValueError("At least 10 valid data points are required for a PSPL plot")

    time, mag, mag_err = time[valid], mag[valid], mag_err[valid]
    order = np.argsort(time)
    time, mag, mag_err = time[order], mag[order], mag_err[order]

    if fit_result is None:
        fit_result = fit_pspl(time, mag, mag_err)
    if fit_result is None:
        raise ValueError("PSPL fitting failed; no plot can be made")

    model_at, time, mag, mag_err = _model_magnitude(time, mag, mag_err, fit_result)
    model_time = np.linspace(time.min(), time.max(), max(2, int(model_points)))
    model_mag = model_at(model_time)
    data_model_mag = model_at(time)

    if ax is None:
        fig, (mag_ax, residual_ax) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10, 6),
            gridspec_kw={"height_ratios": (3, 1)},
            constrained_layout=True,
        )
    else:
        mag_ax = ax
        fig = mag_ax.figure
        residual_ax = mag_ax.figure.add_axes(
            [mag_ax.get_position().x0, 0.12, mag_ax.get_position().width, 0.20],
            sharex=mag_ax,
        )

    # A short-cadence season can contain thousands of points. Draw the full
    # light curve as a line and keep error bars only on a readable subset;
    # plotting every point as a large marker hides the actual shape.
    point_stride = max(1, len(time) // 1000)
    shown = np.arange(0, len(time), point_stride)
    mag_ax.plot(time, mag, lw=0.8, alpha=0.78, label="observed light curve")
    mag_ax.errorbar(
        time[shown],
        mag[shown],
        yerr=mag_err[shown],
        fmt="o",
        ms=3.0,
        alpha=0.55,
        elinewidth=0.7,
        capsize=0,
        label="_nolegend_",
    )
    mag_ax.plot(model_time, model_mag, lw=2.2, label="PSPL fit")
    mag_ax.invert_yaxis()
    mag_ax.set_ylabel("magnitude")
    mag_ax.set_title(title)
    mag_ax.grid(alpha=0.25)
    mag_ax.legend(loc="best")

    residual = mag - data_model_mag
    residual_ax.plot(time, residual, lw=0.8, alpha=0.78)
    residual_ax.errorbar(
        time[shown],
        residual[shown],
        yerr=mag_err[shown],
        fmt="o",
        ms=2.5,
        alpha=0.5,
        elinewidth=0.6,
        capsize=0,
    )
    residual_ax.axhline(0.0, color="black", lw=1.0)
    residual_ax.set_xlabel("time")
    residual_ax.set_ylabel("obs − model")
    residual_ax.grid(alpha=0.25)

    annotation = (
        f"t0 = {fit_result['t0_fit']:.3f}\n"
        f"tE = {fit_result['tE_fit']:.2f} d\n"
        f"u0 = {fit_result['u0_fit']:.4g}\n"
        f"χ²red = {fit_result['chi2_red_pspl']:.2f}"
    )
    mag_ax.text(
        0.02,
        0.03,
        annotation,
        transform=mag_ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.75"},
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, (mag_ax, residual_ax), fit_result
