"""
Thin client for the USGS web services used by the AEGIS app:
site lists (NWIS site service), instantaneous values (NWIS IV),
annual peaks (NWIS peak) and upstream basins (NLDI).
"""
from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd
import requests

NWIS = "https://nwis.waterservices.usgs.gov/nwis"
PEAK = "https://nwis.waterdata.usgs.gov/nwis/peak"
NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite"
SQMI_TO_KM2 = 2.58998811
CFS_TO_CMS = 0.0283168466

# UTC offsets (hours) for the tz_cd codes NWIS returns
TZ_OFFSETS = {
    "UTC": 0, "AST": -4, "ADT": -3, "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6, "PST": -8, "PDT": -7, "AKST": -9, "AKDT": -8,
    "HST": -10, "SST": -11, "ChST": 10, "GST": 10,
}

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "PR": "Puerto Rico", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

_session = requests.Session()
_session.headers["User-Agent"] = "AEGIS-streamlit-app"


def _read_rdb(text: str) -> pd.DataFrame:
    """Parse a USGS RDB (tab-delimited) payload into a string DataFrame."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()
    # second non-comment line holds the column formats (e.g. '5s  15s ...')
    return pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])),
                       sep="\t", dtype=str, keep_default_na=False)


def _get(url: str, params: dict | None = None, timeout: int = 120) -> requests.Response | None:
    """GET that returns None for the 404 NWIS uses to signal 'no data'."""
    r = _session.get(url, params=params, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r


def get_sites(state_cd: str) -> pd.DataFrame:
    """
    Stream gauges in a state with instantaneous discharge (00060).

    Returns one row per site with name, location, drainage area (km2),
    time zone and the period of record of the IV discharge series.
    """
    base = dict(format="rdb", stateCd=state_cd, siteType="ST", hasDataTypeCd="iv",
                parameterCd="00060", siteStatus="all")
    r = _get(f"{NWIS}/site/", {**base, "siteOutput": "expanded"})
    if r is None:
        return pd.DataFrame()
    info = _read_rdb(r.text)
    info = info[["site_no", "station_nm", "dec_lat_va", "dec_long_va",
                 "drain_area_va", "contrib_drain_area_va", "tz_cd", "huc_cd"]]

    r = _get(f"{NWIS}/site/", {**base, "seriesCatalogOutput": "true",
                                "outputDataTypeCd": "iv"})
    cat = _read_rdb(r.text) if r is not None else pd.DataFrame()
    if not cat.empty:
        cat = cat[(cat["parm_cd"] == "00060") & (cat["data_type_cd"] == "uv")]
        cat = cat.groupby("site_no").agg(begin_date=("begin_date", "min"),
                                         end_date=("end_date", "max")).reset_index()
        info = info.merge(cat, on="site_no", how="left")
    else:
        info["begin_date"] = info["end_date"] = ""

    for c in ("dec_lat_va", "dec_long_va", "drain_area_va", "contrib_drain_area_va"):
        info[c] = pd.to_numeric(info[c], errors="coerce")
    for c in ("begin_date", "end_date"):
        info[c] = pd.to_datetime(info[c], errors="coerce")
    info["area_km2"] = info["drain_area_va"] * SQMI_TO_KM2
    info = info.drop_duplicates("site_no").sort_values("site_no").reset_index(drop=True)
    return info


def _pick_column(df: pd.DataFrame, pcode: str) -> pd.Series | None:
    """NWIS may return several time series per parameter; keep the fullest one."""
    cols = [c for c in df.columns if c.endswith(f"_{pcode}")]
    if not cols:
        return None
    values = {c: pd.to_numeric(df[c], errors="coerce") for c in cols}
    best = max(values, key=lambda c: values[c].notna().sum())
    return values[best]


def get_iv(site_no: str, start: date, end: date, tz_cd: str = "UTC",
           chunk_years: int = 2, progress=None) -> pd.DataFrame:
    """
    Instantaneous discharge (cfs) and stage (ft) for a site.

    Requests are split into `chunk_years` blocks. Timestamps are returned
    as naive datetimes in the site's local *standard* time (no DST jumps).
    `progress` is an optional callable(fraction) for UI feedback.
    """
    bounds = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1),
                           freq=pd.DateOffset(years=chunk_years))
    bounds = list(bounds) + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    bounds = sorted(set(bounds))

    frames = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        params = dict(format="rdb", sites=site_no, parameterCd="00060,00065",
                      siteStatus="all", startDT=a.strftime("%Y-%m-%d"),
                      endDT=(b - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        r = _get(f"{NWIS}/iv/", params, timeout=300)
        if r is not None:
            raw = _read_rdb(r.text)
            if not raw.empty:
                offset = raw["tz_cd"].map(TZ_OFFSETS).fillna(0).astype(float)
                utc = pd.to_datetime(raw["datetime"]) - pd.to_timedelta(offset, unit="h")
                q = _pick_column(raw, "00060")
                h = _pick_column(raw, "00065")
                frames.append(pd.DataFrame({
                    "Q_cfs": q.values if q is not None else np.nan,
                    "stage_ft": h.values if h is not None else np.nan,
                }, index=utc))
        if progress:
            progress((i + 1) / (len(bounds) - 1))

    if not frames:
        return pd.DataFrame(columns=["Q_cfs", "stage_ft"])
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated()]
    df.index = df.index + pd.Timedelta(hours=TZ_OFFSETS.get(tz_cd, 0))
    df.index.name = "time"
    # negative discharges (tidal reversals, sensor artifacts) -> missing
    df.loc[df["Q_cfs"] < 0, "Q_cfs"] = np.nan
    return df


def get_annual_peaks(site_no: str) -> pd.Series:
    """Annual peak discharges (cfs) from the NWIS peak-flow file."""
    r = _get(PEAK, dict(site_no=site_no, agency_cd="USGS", format="rdb"), timeout=60)
    if r is None:
        return pd.Series(dtype=float)
    df = _read_rdb(r.text)
    if df.empty:
        return pd.Series(dtype=float)
    vals = pd.to_numeric(df["peak_va"], errors="coerce")
    years = df["peak_dt"].str[:4]
    return pd.Series(vals.values, index=years.values, name="peak_cfs").dropna()


def get_basin(site_no: str) -> dict | None:
    """Upstream basin polygon (GeoJSON FeatureCollection) from NLDI."""
    try:
        r = _get(f"{NLDI}/USGS-{site_no}/basin",
                 dict(simplified="true", splitCatchment="false"), timeout=90)
    except requests.RequestException:
        return None
    if r is None:
        return None
    gj = r.json()
    return gj if gj.get("features") else None


def get_upstream_mainstem(site_no: str, distance_km: int = 1000) -> dict | None:
    """Upstream main-stem flowlines (GeoJSON) from NLDI."""
    try:
        r = _get(f"{NLDI}/USGS-{site_no}/navigation/UM/flowlines",
                 dict(distance=distance_km), timeout=90)
    except requests.RequestException:
        return None
    if r is None:
        return None
    gj = r.json()
    return gj if gj.get("features") else None


def basin_area_km2(geojson: dict) -> float | None:
    """Area of a GeoJSON polygon in km2 (equal-area projection)."""
    try:
        import geopandas as gpd
        gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
        return float(gdf.to_crs("EPSG:6933").area.sum() / 1e6)
    except Exception:
        return None
