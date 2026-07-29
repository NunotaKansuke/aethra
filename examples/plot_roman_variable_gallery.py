"""Plot one F146 light curve from each roman_variable variability class."""

from pathlib import Path

import numpy as np

from aethra import load_roman_variable

SAMPLES = {
    "CEP": "RGES_filters_CEP_lightcurves/RGES_filters_OGLE-BLG-CEP-019_lightcurves.fits",
    "DSCT": "RGES_filters_DSCT_lightcurves/RGES_filters_OGLE-BLG-DSCT-00697_lightcurves.fits",
    "ECL": "RGES_filters_ECL_lightcurves/RGES_filters_OGLE-BLG-ECL-000013_lightcurves.fits",
    "ELL": "RGES_filters_ELL_lightcurves/RGES_filters_OGLE-BLG-ELL-000008_lightcurves.fits",
    "FL": "RGES_filters_FL_lightcurves/RGES_filters_FL_931_234290432_722_WFmag_24_lightcurves.fits",
    "HB": "RGES_filters_HB_lightcurves/RGES_filters_OGLE-BLG-HB-0011_lightcurves.fits",
    "LPV": "RGES_filters_LPV_lightcurves/RGES_filters_OGLE-BLG-LPV-046129_lightcurves.fits",
    "RRLYR": "RGES_filters_RRLYR_lightcurves/RGES_filters_OGLE-BLG-RRLYR-03103_lightcurves.fits",
    "T2CEP": "RGES_filters_T2CEP_lightcurves/RGES_filters_OGLE-BLG-T2CEP-0066_lightcurves.fits",
}


def plot_gallery(dataset_dir="../roman_variable", output_path="roman-variable-gallery.png"):
    """Save a 3x3 gallery of the first 70 days of representative F146 curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Install the plotting extra: pip install -e '.[plotting]'") from exc

    dataset_dir = Path(dataset_dir)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)

    for ax, (kind, relative_path) in zip(axes.flat, SAMPLES.items()):
        fits_path = dataset_dir / relative_path
        lc = load_roman_variable(fits_path, filters="F146")
        finite = np.isfinite(lc["bjd"]) & np.isfinite(lc["mag"])
        lc = lc.loc[finite].sort_values("bjd")
        time = lc["bjd"].to_numpy()
        mag = lc["mag"].to_numpy()
        in_window = time <= time.min() + 70.0
        time = time[in_window]
        mag = mag[in_window]
        time = time - time.min()

        stride = max(1, len(time) // 1000)
        ax.plot(time, mag, lw=0.8, alpha=0.8)
        ax.plot(time[::stride], mag[::stride], "o", ms=2.5, alpha=0.45)
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
        ax.set_title(f"{kind}: {lc['name'].iloc[0]}")
        ax.set_xlabel("days from season start")
        ax.set_ylabel("magnitude")

    fig.suptitle("roman_variable — representative F146 light curves (first 70 days)")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    plot_gallery()
