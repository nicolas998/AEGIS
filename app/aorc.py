"""
Hourly basin-average precipitation from NOAA AORC v1.1 (1 km, CONUS),
read anonymously from the yearly Zarr stores on AWS.
"""
from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

BUCKET = "noaa-nws-aorc-v1-1-1km"
STORAGE = {"anon": True}
MAX_BLOCK_VALUES = 25_000_000      # cells x hours loaded at once (~200 MB float64)


def available_years() -> list[int]:
    """Years published in the AORC bucket (one Zarr store per year)."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    return sorted(int(p.rsplit("/", 1)[-1].split(".")[0])
                  for p in fs.ls(BUCKET) if p.endswith(".zarr"))


def last_date() -> date:
    """Last day covered by AORC (end of the most recent published year)."""
    try:
        return date(available_years()[-1], 12, 31)
    except Exception:
        return date(2025, 12, 31)


def _basin_mask(basin: dict, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Grid cells whose centres fall inside the basin (nearest cell if none)."""
    import geopandas as gpd
    from shapely import contains_xy
    poly = gpd.GeoDataFrame.from_features(basin["features"], crs="EPSG:4326").union_all()
    lon2, lat2 = np.meshgrid(lon, lat)
    mask = contains_xy(poly, lon2, lat2)
    if not mask.any():
        c = poly.centroid
        mask[np.abs(lat - c.y).argmin(), np.abs(lon - c.x).argmin()] = True
    return mask


def basin_rainfall(basin: dict, start: date, end: date, utc_offset_h: float = 0,
                   progress=None) -> pd.Series:
    """
    Hourly basin-average precipitation (mm/h) between `start` and `end`.

    Times are returned in local standard time (UTC + `utc_offset_h`), naive,
    matching the USGS series. Hours after the last AORC year are not returned.
    `progress` is an optional callable(fraction).
    """
    import geopandas as gpd
    x0, y0, x1, y1 = gpd.GeoDataFrame.from_features(
        basin["features"], crs="EPSG:4326").total_bounds
    pad = 0.01
    offset = pd.Timedelta(hours=utc_offset_h)
    t0 = pd.Timestamp(start) - offset
    t1 = pd.Timestamp(end) + pd.Timedelta(hours=23) - offset

    published = set(available_years())
    years = [y for y in range(t0.year, t1.year + 1) if y in published]
    parts, mask = [], None
    for i, year in enumerate(years):
        da = xr.open_zarr(f"s3://{BUCKET}/{year}.zarr", storage_options=STORAGE)["APCP_surface"]
        da = da.sel(latitude=slice(y0 - pad, y1 + pad), longitude=slice(x0 - pad, x1 + pad),
                    time=slice(max(t0, pd.Timestamp(year, 1, 1)),
                               min(t1, pd.Timestamp(year, 12, 31, 23))))
        if da.sizes["time"] == 0:
            continue
        if mask is None:
            mask = _basin_mask(basin, da.longitude.values, da.latitude.values)
        # load in time blocks so large basins stay within memory
        step = max(24, MAX_BLOCK_VALUES // mask.size)
        for k in range(0, da.sizes["time"], step):
            block = da.isel(time=slice(k, k + step)).values      # (time, lat, lon)
            with warnings.catch_warnings():     # hours missing in AORC stay NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(block[:, mask], axis=1)
            parts.append(pd.Series(mean, index=da.time.values[k:k + step]))
        if progress:
            progress((i + 1) / len(years))

    if not parts:
        return pd.Series(dtype=float, name="rain_mm")
    rain = pd.concat(parts).sort_index()
    rain.index = pd.DatetimeIndex(rain.index) + offset
    rain.index.name = "time"
    return rain.rename("rain_mm")      # kg/m2 per hour == mm/h
