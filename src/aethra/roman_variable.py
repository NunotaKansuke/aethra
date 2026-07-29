"""Loader for the RGES ``roman_variable`` FITS light curves.

The dataset stores one filter per FITS table extension and stores the time
arrays separately under ``top_level``. This module turns one such FITS file
into the tidy table expected by :func:`aethra.run_pipeline_from_dataframe`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_roman_variable"]


_TIME_FILES = {
    "F087": "roman_times_longcadence.npy",
    "F146": "roman_times_shortcadence.npy",
    "F213": "roman_times_longcadence2.npy",
}


def _infer_time_dir(fits_path):
    """Find the dataset ``top_level`` directory above a FITS file."""
    fits_path = Path(fits_path).resolve()
    for parent in fits_path.parents:
        candidate = parent / "top_level"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find roman_variable/top_level above "
        f"{str(fits_path)!r}; pass time_dir explicitly."
    )


def _column_name(table, wanted):
    """Return a table column name case-insensitively."""
    names = {str(name).lower(): name for name in table.columns.names}
    try:
        return names[wanted.lower()]
    except KeyError as exc:
        raise ValueError(
            f"FITS extension {table.name!r} has no '{wanted}' column; "
            f"available columns: {table.columns.names}"
        ) from exc


def _normalise_filters(filters):
    if filters is None:
        return None
    if isinstance(filters, str):
        return {filters.upper()}
    return {str(value).upper() for value in filters}


def load_roman_variable(fits_path, time_dir=None, filters=None):
    """Load one RGES ``roman_variable`` FITS light curve.

    Parameters
    ----------
    fits_path : str or pathlib.Path
        Path to one ``*_lightcurves.fits`` file.
    time_dir : str or pathlib.Path, optional
        Directory containing ``roman_times_*.npy``. If omitted, the loader
        searches parent directories for ``top_level``; this works directly
        with the checked-out ``roman_variable`` dataset.
    filters : str or iterable of str, optional
        Filters to load. By default all supported extensions are loaded.

    Returns
    -------
    pandas.DataFrame
        Columns are ``bjd``, ``mag``, ``mag_err``, ``filt``, and ``name``.
        Rows are sorted by time and filter.

    Notes
    -----
    The time mapping is the dataset convention: F087 uses long cadence,
    F146 uses short cadence, and F213 uses the second long-cadence array.
    """
    fits_path = Path(fits_path)
    if not fits_path.is_file():
        raise FileNotFoundError(f"FITS file not found: {str(fits_path)!r}")
    if fits_path.suffix.lower() not in {".fits", ".fit"}:
        raise ValueError(f"Expected a FITS file, got {str(fits_path)!r}")

    selected_filters = _normalise_filters(filters)
    if selected_filters is not None:
        unknown = selected_filters - set(_TIME_FILES)
        if unknown:
            raise ValueError(
                f"Unsupported roman_variable filter(s): {sorted(unknown)}; "
                f"supported filters: {sorted(_TIME_FILES)}"
            )

    if time_dir is None:
        time_dir = _infer_time_dir(fits_path)
    else:
        time_dir = Path(time_dir)
    if not time_dir.is_dir():
        raise FileNotFoundError(f"Time directory not found: {str(time_dir)!r}")

    from astropy.io import fits

    pieces = []
    with fits.open(fits_path, memmap=True) as hdul:
        fallback_name = fits_path.stem
        object_name = str(hdul[0].header.get("NAME", fallback_name)).strip()

        for hdu in hdul[1:]:
            filt = str(hdu.header.get("EXTNAME", "")).strip().upper()
            if filt not in _TIME_FILES:
                continue
            if selected_filters is not None and filt not in selected_filters:
                continue
            if hdu.data is None or hdu.columns is None:
                continue

            mag_col = _column_name(hdu, "mag")
            err_col = _column_name(hdu, "mag_error")
            time_path = time_dir / _TIME_FILES[filt]
            if not time_path.is_file():
                raise FileNotFoundError(
                    f"Missing time array for {filt}: {str(time_path)!r}"
                )

            time = np.load(time_path, allow_pickle=False)
            mag = np.array(hdu.data[mag_col], dtype=float, copy=True)
            mag_err = np.array(hdu.data[err_col], dtype=float, copy=True)
            if len(time) != len(mag):
                raise ValueError(
                    f"Length mismatch for {filt} in {str(fits_path)!r}: "
                    f"{len(time)} times vs {len(mag)} photometric rows"
                )

            pieces.append(
                pd.DataFrame(
                    {
                        "bjd": np.array(time, dtype=float, copy=True),
                        "mag": mag,
                        "mag_err": mag_err,
                        "filt": filt,
                        "name": object_name,
                    }
                )
            )

    if not pieces:
        raise ValueError(
            f"No supported roman_variable filter extensions found in "
            f"{str(fits_path)!r}; supported filters: {sorted(_TIME_FILES)}"
        )

    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["bjd", "filt"], kind="stable")
        .reset_index(drop=True)
    )
