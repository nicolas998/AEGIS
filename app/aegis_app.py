"""
AEGIS event detection for USGS gauges.

Run with:  streamlit run app/aegis_app.py
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import aorc
import pipeline as pl
import storage
import usgs

st.set_page_config(page_title="AEGIS event detection", page_icon="🌊", layout="wide")

# ---------------------------------------------------------------- styling
DARK = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
C = dict(
    discharge="#3987e5" if DARK else "#2a78d6",
    baseflow="#d95926" if DARK else "#eb6834",
    stage="#199e70" if DARK else "#1baf7a",
    selected="#9085e9" if DARK else "#6250d6",
    rain="#8d99a8" if DARK else "#6b7889",
    band="rgba(160,160,160,0.16)",
    selected_band="rgba(98,80,214,0.20)",
    muted="#c3c2b7" if DARK else "#52514e",
    surface="#0e1117" if DARK else "#ffffff",
)
MAP_STYLE = "carto-darkmatter" if DARK else "carto-positron"


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    layout = dict(height=height, margin=dict(l=10, r=10, t=30, b=10),
                  hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0))
    fig.update_layout(**{**layout, **kw})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


# ---------------------------------------------------------------- cached data access
@st.cache_data(ttl=86400, show_spinner=False)
def load_sites(state_cd: str) -> pd.DataFrame:
    return usgs.get_sites(state_cd)


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def load_basin(site_no: str):
    return usgs.get_basin(site_no), usgs.get_upstream_mainstem(site_no)


def _complete_on_disk(path: Path, end: date) -> bool:
    """A saved file is reused only if it was written after the period ended."""
    return path.exists() and date.fromtimestamp(path.stat().st_mtime) > end


@st.cache_data(ttl=86400, show_spinner=False, max_entries=10)
def load_iv(site_no: str, start: date, end: date, tz_cd: str) -> pd.DataFrame:
    path = storage.iv_path(site_no, start, end)
    if _complete_on_disk(path, end):
        return storage.read_table(path)
    iv = usgs.get_iv(site_no, start, end, tz_cd)
    if not iv.empty:
        storage.write_table(iv, path, f"time_{tz_cd}")
    return iv


@st.cache_data(ttl=86400, show_spinner=False)
def aorc_last_date() -> date:
    return aorc.last_date()


@st.cache_data(ttl=86400, show_spinner=False, max_entries=10)
def load_rain(site_no: str, start: date, end: date, tz_cd: str) -> pd.Series | None:
    """Hourly AORC basin-average rainfall (mm/h), or None without a basin."""
    path = storage.rain_path(site_no, start, end)
    if _complete_on_disk(path, min(end, aorc_last_date())):
        return storage.read_table(path)["rain_mm"]
    basin, _ = load_basin(site_no)
    if not basin:
        return None
    rain = aorc.basin_rainfall(basin, start, end, usgs.TZ_OFFSETS.get(tz_cd, 0))
    if not rain.empty:
        storage.write_table(rain, path, f"time_{tz_cd}")
    return rain


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def load_peaks(site_no: str) -> pd.Series:
    return usgs.get_annual_peaks(site_no)


@st.cache_data(show_spinner=False, max_entries=20)
def analyze(site_no, start, end, tz_cd, area_km2, params: pl.Params, use_peaks: bool):
    iv = load_iv(site_no, start, end, tz_cd)
    if iv.empty or iv["Q_cfs"].notna().sum() == 0:
        return None
    peaks = load_peaks(site_no) if use_peaks else None
    return pl.run_aegis(iv, params, area_km2, peaks)


# ---------------------------------------------------------------- figures
def _lines_from_geojson(gj: dict):
    """Flatten (Multi)LineString features into lon/lat lists with None breaks."""
    lon, lat = [], []
    for f in gj["features"]:
        g = f["geometry"]
        parts = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        for part in parts:
            lon += [c[0] for c in part] + [None]
            lat += [c[1] for c in part] + [None]
    return lon, lat


def map_figure(site: pd.Series, others: pd.DataFrame, basin, mainstem) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lon=others["dec_long_va"], lat=others["dec_lat_va"], mode="markers",
        marker=dict(size=8, color="rgba(128,128,128,0.6)"),
        # keep the look unchanged when plotly marks a clicked point as selected
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
        customdata=others["site_no"],
        text=others["site_no"] + " · " + others["station_nm"],
        hovertemplate="%{text}<br><i>click to select</i><extra></extra>",
        name="Other gauges"))
    if mainstem:
        lon, lat = _lines_from_geojson(mainstem)
        fig.add_trace(go.Scattermap(lon=lon, lat=lat, mode="lines",
                                    line=dict(width=2, color=C["discharge"]),
                                    hoverinfo="skip", name="Upstream main stem"))
    fig.add_trace(go.Scattermap(
        lon=[site.dec_long_va], lat=[site.dec_lat_va], mode="markers",
        marker=dict(size=15, color=C["baseflow"]),
        text=[f"{site.site_no} · {site.station_nm}"],
        hovertemplate="%{text}<extra></extra>", name="Selected gauge"))

    layers, zoom = [], 9.0
    center = dict(lon=site.dec_long_va, lat=site.dec_lat_va)
    if basin:
        layers = [
            dict(sourcetype="geojson", source=basin, type="fill", below="traces",
                 color="rgba(42,120,214,0.18)"),
            dict(sourcetype="geojson", source=basin, type="line", below="traces",
                 color=C["discharge"], line=dict(width=1.5)),
        ]
        lon, lat = [], []
        for f in basin["features"]:
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for poly in polys:
                lon += [c[0] for c in poly[0]]
                lat += [c[1] for c in poly[0]]
        center = dict(lon=(min(lon) + max(lon)) / 2, lat=(min(lat) + max(lat)) / 2)
        span = max(max(lon) - min(lon), (max(lat) - min(lat)) * 1.4, 0.01)
        zoom = float(np.clip(math.log2(360 / span) - 0.7, 2, 13))

    fig.update_layout(map=dict(style=MAP_STYLE, center=center, zoom=zoom, layers=layers),
                      height=380, margin=dict(l=0, r=0, t=0, b=0), showlegend=True,
                      clickmode="event+select", hovermode="closest",
                      legend=dict(orientation="h", y=0.01, x=0.01,
                                  bgcolor="rgba(128,128,128,0.12)"))
    return fig


def _event_bands(events: pd.DataFrame, top: float):
    """Rectangles for all events as one filled trace (one polygon per event)."""
    x, y = [], []
    for s, e in zip(events["start"], events["end"]):
        x += [s, s, e, e, s, None]
        y += [0, top, top, 0, 0, None]
    return x, y


def _with_rain_panel(rain: pd.Series | None, **kw):
    """Subplots with an inverted rainfall panel on top when rain is given.
    Returns the figure and the row that holds the hydrograph."""
    if rain is None or rain.empty:
        return make_subplots(rows=1, cols=1, **kw), 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.24, 0.76], **kw)
    return fig, 2


def _add_rain(fig: go.Figure, rain: pd.Series, label: str, unit: str) -> None:
    hover = f"%{{y:.1f}} {unit}<extra>Rainfall</extra>"
    if len(rain) > 400:
        # sub-pixel bars drop out when rendered; a filled step trace always shows
        trace = go.Scatter(x=rain.index, y=rain.fillna(0).values, name=label,
                           mode="lines", line=dict(width=1, color=C["rain"], shape="hv"),
                           fill="tozeroy", fillcolor=C["rain"], hovertemplate=hover)
    else:
        trace = go.Bar(x=rain.index, y=rain.values, name=label,
                       marker=dict(color=C["rain"], line=dict(width=0)),
                       hovertemplate=hover)
    fig.add_trace(trace, row=1, col=1)
    fig.update_yaxes(title_text=f"Rain ({unit})", autorange="reversed", row=1, col=1)


def daily_figure(res: pl.Result, params: pl.Params, sel: pd.Series | None,
                 rain: pd.Series | None = None) -> go.Figure:
    daily = res.hourly.resample("D").agg({"qz": "max", "baseflow": "mean", "Q_cfs": "max"})
    top = float(max(np.nanmax(daily["qz"]), params.event_threshold, 1.0) * 1.08)
    fig, r = _with_rain_panel(rain)
    if r == 2:
        _add_rain(fig, rain.resample("D").sum(min_count=1), "Rainfall (daily, AORC)", "mm/d")
    bx, by = _event_bands(res.events, top)
    fig.add_trace(go.Scatter(x=bx, y=by, fill="toself", fillcolor=C["band"],
                             line=dict(width=0), mode="lines", hoverinfo="skip",
                             name="Detected events"), row=r, col=1)
    if sel is not None:
        fig.add_trace(go.Scatter(
            x=[sel.start, sel.start, sel.end, sel.end, sel.start], y=[0, top, top, 0, 0],
            fill="toself", fillcolor=C["selected_band"], line=dict(width=1, color=C["selected"]),
            mode="lines", hoverinfo="skip", name=f"Event {sel.event}"), row=r, col=1)
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["qz"], mode="lines", name="Discharge (daily max)",
        line=dict(width=1.5, color=C["discharge"]), customdata=daily["Q_cfs"],
        hovertemplate="%{y:.3f} · %{customdata:,.0f} cfs<extra>Discharge</extra>"),
        row=r, col=1)
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["baseflow"], mode="lines", name="Baseflow (daily mean)",
        line=dict(width=1.5, color=C["baseflow"], dash="dot"),
        hovertemplate="%{y:.3f}<extra>Baseflow</extra>"), row=r, col=1)
    fig.add_trace(go.Scatter(
        x=res.events["peak_time"], y=res.events["peak_qz"], mode="markers",
        marker=dict(size=8, color=C["discharge"], line=dict(width=2, color=C["surface"])),
        customdata=res.events["event"], name="Event peaks",
        hovertemplate="Event %{customdata} peak: %{y:.3f}<extra></extra>"), row=r, col=1)
    fig.add_hline(y=params.event_threshold, line=dict(color=C["muted"], width=1, dash="dash"),
                  annotation_text=f"event threshold {params.event_threshold:g}",
                  annotation_position="bottom right", annotation_font_color=C["muted"],
                  row=r, col=1)
    fig.add_hline(y=1.0, line=dict(color=C["muted"], width=1, dash="dot"),
                  annotation_text="mean annual max", annotation_position="top right",
                  annotation_font_color=C["muted"], row=r, col=1)
    fig.update_yaxes(title_text="Q / mean annual max (–)", range=[0, top], row=r, col=1)
    fig.update_xaxes(range=[res.hourly.index[0], res.hourly.index[-1]])
    return _layout(fig, 380 if r == 1 else 480, bargap=0)


def timeline_figure(res: pl.Result, sel_idx: int) -> go.Figure:
    ev = res.events
    t0, t1 = res.hourly.index[0], res.hourly.index[-1]
    min_w = (t1 - t0) / pd.Timedelta(milliseconds=1) * 0.004
    widths = np.maximum((ev["end"] - ev["start"]) / pd.Timedelta(milliseconds=1), min_w)
    colors = [C["selected"] if i == sel_idx else C["discharge"] for i in range(len(ev))]
    fig = go.Figure(go.Bar(
        x=ev["start"], y=ev["peak_qz"], width=widths, offset=0,
        marker=dict(color=colors, line=dict(width=1, color=C["surface"]),
                    cornerradius=4),
        customdata=np.stack([ev["event"], ev["duration_h"]], axis=1),
        hovertemplate=("Event %{customdata[0]}<br>start %{x|%Y-%m-%d %H:%M}"
                       "<br>peak %{y:.3f} · %{customdata[1]:.0f} h<extra></extra>"),
    ))
    fig.update_yaxes(title_text="Peak (–)", rangemode="tozero")
    fig.update_xaxes(range=[t0, t1])
    return _layout(fig, 170, hovermode="closest", showlegend=False, bargap=0,
                   dragmode=False, clickmode="event+select")


def event_figure(res: pl.Result, iv: pd.DataFrame, ev: pd.Series,
                 rain: pd.Series | None = None) -> go.Figure:
    pad = max((ev.end - ev.start) * 0.1, pd.Timedelta(hours=6))
    a, b = ev.start - pad, ev.end + pad
    raw = iv.loc[a:b]
    hr = res.hourly.loc[a:b]
    ev_rain = rain.loc[a:b] if rain is not None else None
    has_rain = ev_rain is not None and not ev_rain.empty
    has_stage = raw["stage_ft"].notna().any()
    heights = ([0.2] if has_rain else []) + [0.55 if has_stage else 0.8] + \
              ([0.25] if has_stage else [])
    fig = make_subplots(rows=len(heights), cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=heights)
    rq = 2 if has_rain else 1
    if has_rain:
        _add_rain(fig, ev_rain, "Rainfall (hourly, AORC)", "mm/h")
    fig.add_trace(go.Scatter(
        x=raw.index, y=raw["Q_cfs"], mode="lines", name="Discharge (instantaneous)",
        line=dict(width=2, color=C["discharge"]),
        hovertemplate="%{y:,.0f} cfs<extra>Discharge</extra>"), row=rq, col=1)
    fig.add_trace(go.Scatter(
        x=hr.index, y=hr["baseflow"] * res.qmax_ann, mode="lines", name="Baseflow",
        line=dict(width=2, color=C["baseflow"], dash="dot"),
        hovertemplate="%{y:,.0f} cfs<extra>Baseflow</extra>"), row=rq, col=1)
    fig.add_trace(go.Scatter(
        x=[ev.peak_time], y=[ev.peak_cfs], mode="markers+text", name="Peak",
        marker=dict(size=10, color=C["discharge"], line=dict(width=2, color=C["surface"])),
        text=[f"{ev.peak_cfs:,.0f} cfs"], textposition="top right",
        textfont=dict(color=C["muted"]), hoverinfo="skip", showlegend=False),
        row=rq, col=1)
    fig.update_yaxes(title_text="Discharge (cfs)", rangemode="tozero", row=rq, col=1)
    if has_stage:
        fig.add_trace(go.Scatter(
            x=raw.index, y=raw["stage_ft"], mode="lines", name="Stage",
            line=dict(width=2, color=C["stage"]),
            hovertemplate="%{y:.2f} ft<extra>Stage</extra>"), row=rq + 1, col=1)
        fig.update_yaxes(title_text="Stage (ft)", row=rq + 1, col=1)
    # after the traces: row="all" skips subplots that are still empty
    fig.add_vrect(x0=ev.start, x1=ev.end, fillcolor=C["selected_band"], line_width=0,
                  layer="below", row="all", col=1)
    height = 360 + (120 if has_stage else 0) + (110 if has_rain else 0)
    return _layout(fig, height, bargap=0)


# ---------------------------------------------------------------- helpers
def fmt_hours(h: float) -> str:
    return f"{h:.0f} h" if h < 72 else f"{h:.0f} h ({h / 24:.1f} d)"


def event_label(ev: pd.Series) -> str:
    return f"Event {ev.event} · {ev.peak_time:%Y-%m-%d} · peak {ev.peak_qz:.2f}"


# ================================================================ sidebar
ss = st.session_state
with st.sidebar:
    st.header("AEGIS")
    st.caption("Automated Event & HydroGraph Identification Strategy")

    st.subheader("1 · Gauge")
    states = list(usgs.STATES)
    state = st.selectbox("State", states, index=states.index("FL"),
                         format_func=lambda s: usgs.STATES[s])
    try:
        with st.spinner("Loading gauges…"):
            sites = load_sites(state)
    except Exception as exc:
        st.error(f"Could not load the USGS site list: {exc}")
        st.stop()
    if sites.empty:
        st.warning("No stream gauges with instantaneous discharge in this state.")
        st.stop()
    names = dict(zip(sites["site_no"], sites["station_nm"]))
    # a gauge clicked on the map (previous run) becomes the selected gauge
    map_state = ss.get(ss.map_key) if "map_key" in ss else None
    points = map_state.selection.points if map_state and map_state.selection else []
    clicked = next((p.get("customdata") for p in points if p.get("customdata") in names),
                   None)
    if clicked:
        ss.site_select = clicked
    elif ss.get("site_select") not in names:
        ss.pop("site_select", None)
    site_no = st.selectbox("USGS gauge", sites["site_no"].tolist(), key="site_select",
                           format_func=lambda s: f"{s} · {names[s]}",
                           help="Gauges with instantaneous discharge (parameter 00060). "
                                "Type to search by number or name, or click a gauge "
                                "on the map.")
    site = sites.set_index("site_no").loc[site_no]
    site["site_no"] = site_no

    st.subheader("2 · Period")
    rec_start = site.begin_date.date() if pd.notna(site.begin_date) else date(2007, 10, 1)
    rec_end = min(site.end_date.date() if pd.notna(site.end_date) else date.today(),
                  date.today())
    default_start = max(rec_start, rec_end - timedelta(days=5 * 365))
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", default_start, min_value=rec_start, max_value=rec_end,
                          key=f"start_{site_no}")
    end = c2.date_input("End", rec_end, min_value=rec_start, max_value=rec_end,
                        key=f"end_{site_no}")

    st.subheader("3 · Baseflow separation")
    filter_name = st.selectbox("Digital filter", list(pl.FILTERS),
                               help="Nathan & McMahon is the AEGIS default: it separated "
                                    "baseflow with one parameter set across all tested "
                                    "watersheds.")
    fparam = st.number_input("Filter parameter α", 0.5, 0.9999, 0.995, 0.001,
                             format="%.4f",
                             help="Recession constant of the filter (0–1), applied to "
                                  "hourly data.")
    bfi = 0.8
    if pl.FILTERS[filter_name] == "Eckhardt":
        bfi = st.number_input("BFImax", 0.01, 0.99, 0.8, 0.05,
                              help="Long-term ratio of baseflow to total streamflow.")

    st.subheader("4 · Event selection")
    event_threshold = st.slider(
        "Event threshold", 0.0, 1.0, 0.2, 0.05,
        help="Minimum event peak relative to the mean annual maximum flow "
             "(1 = mean annual flood). Events whose peak does not exceed it are discarded.")
    separation = pl.SEPARATIONS[st.radio(
        "Event separation", list(pl.SEPARATIONS),
        help="**Peak-based split** cuts the runoff windows at the trough between "
             "peaks, so storms that follow each other closely are kept as separate "
             "events. **AEGIS original** keeps one event per start-to-start runoff "
             "window (only its largest peak counts).")]
    min_rise, smooth_hours = 0.1, 0
    if separation == "peaks":
        min_rise = st.number_input(
            "Minimum rise", 0.01, 1.0, 0.1, 0.01,
            help="A peak is a separate event when it rises at least this much "
                 "(× mean annual max) above the trough separating it from its neighbour.")
        smooth_hours = st.number_input(
            "Smoothing window (h)", 0, 72, 0, 1,
            help="Moving average used only to locate peaks and troughs. Use ~25 h for "
                 "tidally affected or noisy gauges; keep 0 for flashy rivers.")

    with st.expander("Advanced"):
        runoff_threshold = st.number_input(
            "Runoff threshold", 0.0, 0.5, 0.01, 0.005, format="%.3f",
            help="Normalized runoff below this value is set to zero before binarization.")
        min_peakness = st.slider(
            "Minimum peakness", 0.0, 1.0, 0.2 if separation == "peaks" else 0.6, 0.05,
            key=f"peakness_{separation}",
            help="(peak − start) / peak must exceed this value (event prominence). "
                 "Default 0.6 for AEGIS original, 0.2 for the peak-based split.")
        max_nans = st.slider("Max missing fraction", 0.0, 1.0, 0.2, 0.05,
                             help="Maximum fraction of missing hourly values in an event.")
        norm_source = st.radio(
            "Mean annual maximum from",
            ["Selected period (AEGIS)", "USGS annual peak record"],
            help="AEGIS normalizes by the mean of the annual maxima of the hourly record. "
                 "For short periods the USGS annual-peak file gives a more stable value.")
        use_ttp = st.checkbox("Time-to-peak clean-up", False,
                              help="Keep only events whose time-to-peak is below a "
                                   "percentile; removes long multi-peak events. It always "
                                   "drops the slowest-rising share of events, including "
                                   "large ones.")
        ttp_pct = st.slider("Time-to-peak percentile", 10, 100, 50, 5, disabled=not use_ttp)

    params = pl.Params(
        filter_name=filter_name, fparam=float(fparam), bfi=float(bfi),
        event_threshold=float(event_threshold), runoff_threshold=float(runoff_threshold),
        max_nans=float(max_nans), min_peakness=float(min_peakness),
        ttp_percentile=float(ttp_pct) if use_ttp else None,
        separation=separation, min_rise=float(min_rise), smooth_hours=int(smooth_hours))

    st.subheader("5 · Rainfall")
    aorc_end = aorc_last_date()
    with_rain = st.checkbox(
        "Add AORC basin rainfall", False,
        help="Hourly basin-average precipitation from NOAA AORC v1.1 (1 km, CONUS), "
             "averaged over the NLDI upstream basin.")
    if with_rain:
        st.caption(f"⏱️ Increases processing time: a few seconds per year for small "
                   f"basins, minutes for large ones. AORC is available through "
                   f"**{aorc_end:%Y-%m-%d}**; no rainfall after that date.")
        if end > aorc_end:
            st.warning(f"The period ends after {aorc_end:%Y-%m-%d}: rainfall will be "
                       "missing for the last part.")

    area = site.area_km2 if pd.notna(site.area_km2) else None
    config = dict(site_no=site_no, start=start, end=end, tz_cd=site.tz_cd or "UTC",
                  area_km2=area, params=params,
                  use_peaks=norm_source.startswith("USGS"), with_rain=with_rain)

    if start >= end:
        st.error("Start date must be before the end date.")
    elif st.button("Detect events", type="primary", width="stretch"):
        ss.run = config

# ================================================================ gauge context
st.title("Runoff event detection for USGS gauges")

basin, mainstem = load_basin(site_no)
basin_area = usgs.basin_area_km2(basin) if basin else None

m_col, i_col = st.columns([3, 2], gap="large")
with m_col:
    # a fresh key per gauge, so a past click never lingers as a stale selection
    ss.map_key = f"gauge_map_{state}_{site_no}"
    st.plotly_chart(map_figure(site, sites[sites["site_no"] != site_no], basin, mainstem),
                    key=ss.map_key, on_select="rerun", selection_mode="points",
                    width="stretch", config=dict(scrollZoom=True, displaylogo=False))
    st.caption("Click a gray gauge on the map to select it.")
with i_col:
    st.subheader(site.station_nm)
    st.markdown(f"USGS [{site_no}](https://waterdata.usgs.gov/monitoring-location/"
                f"USGS-{site_no}/) · HUC {site.huc_cd}")
    a1, a2 = st.columns(2)
    a1.metric("Upstream area (USGS)",
              f"{site.area_km2:,.0f} km²" if pd.notna(site.area_km2) else "n/a")
    a2.metric("Basin polygon (NLDI)", f"{basin_area:,.0f} km²" if basin_area else "n/a")
    st.metric("Instantaneous discharge record",
              f"{rec_start:%Y-%m-%d} → {rec_end:%Y-%m-%d}")
    if not basin:
        st.caption("No NLDI basin is available for this gauge.")

# ================================================================ results
run = ss.get("run")
if not run or run["site_no"] != site_no:
    st.info("Pick the period and parameters in the sidebar, then press **Detect events**.")
    st.stop()
if run != config:
    st.caption("⚠️ Settings changed since the last run — press **Detect events** to update.")

if run["area_km2"] is None and basin_area:
    run = {**run, "area_km2": basin_area}
try:
    with st.spinner("Downloading instantaneous discharge and stage from USGS and "
                    "running AEGIS… (multi-year requests can take a minute)"):
        res = analyze(**{k: v for k, v in run.items() if k != "with_rain"})
        iv = load_iv(run["site_no"], run["start"], run["end"], run["tz_cd"])
except Exception as exc:
    st.error(f"Processing failed: {exc}")
    st.stop()
if res is None:
    st.warning("USGS returned no instantaneous discharge for this gauge and period.")
    st.stop()

p: pl.Params = run["params"]

rain = None
if run.get("with_rain"):
    if not basin:
        st.warning("No NLDI basin is available for this gauge, so AORC rainfall cannot "
                   "be averaged over it.")
    elif run["start"] > aorc_end:
        st.warning(f"AORC rainfall is available only through {aorc_end:%Y-%m-%d}; the "
                   "selected period starts after that.")
    else:
        try:
            with st.spinner("Extracting AORC basin rainfall… (adds processing time, "
                            "especially for long periods or large basins)"):
                rain = load_rain(run["site_no"], run["start"], run["end"], run["tz_cd"])
        except Exception as exc:
            st.warning(f"AORC rainfall could not be loaded: {exc}")
        if run["end"] > aorc_end:
            st.info(f"AORC rainfall is available only through {aorc_end:%Y-%m-%d}; "
                    "no rainfall is shown after that date.")
ev_tab = pl.add_rainfall(res.events, rain) if rain is not None else res.events

# per-gauge folder: data, basin and the events of this run
iv_file = storage.iv_path(run["site_no"], run["start"], run["end"])
rain_file = storage.rain_path(run["site_no"], run["start"], run["end"])
ev_file = storage.events_path(run["site_no"], run["start"], run["end"])
try:
    if not iv_file.exists():
        storage.write_table(iv, iv_file, f"time_{run['tz_cd']}")
    if rain is not None and not rain_file.exists():
        storage.write_table(rain, rain_file, f"time_{run['tz_cd']}")
    storage.write_basin(run["site_no"], basin)
except OSError as exc:
    st.warning(f"Could not save data files: {exc}")

# record the configuration on screen so it can be inspected outside the browser
last_run = dict(
    written_at=datetime.now().isoformat(timespec="seconds"),
    state=state, site_no=run["site_no"], station_nm=names[run["site_no"]],
    start=str(run["start"]), end=str(run["end"]), tz_cd=run["tz_cd"],
    area_km2=run["area_km2"], params=asdict(p),
    normalization="USGS annual peaks" if run["use_peaks"] else "selected period",
    aorc_rainfall=bool(run.get("with_rain")),
    rain_hours=0 if rain is None else len(rain),
    data_folder=str(storage.gauge_dir(run["site_no"])),
    settings_changed_since_run=run != config,
    results=dict(n_events=len(res.events), n_detected=res.n_detected,
                 qmax_ann_cfs=res.qmax_ann, n_years=res.n_years, s4=res.s4,
                 completeness=res.completeness, warnings=res.warnings),
)
try:
    (Path(__file__).parent / "last_run.json").write_text(json.dumps(last_run, indent=2))
    storage.write_events(ev_tab, ev_file, last_run)
except OSError:
    pass
st.divider()
k = st.columns(4)
k[0].metric("Events", f"{len(ev_tab)}",
            f"of {res.n_detected} detected" if p.ttp_percentile else None,
            delta_color="off")
k[1].metric("Mean annual maximum", f"{res.qmax_ann:,.0f} cfs",
            f"{res.n_years} years", delta_color="off",
            help="Normalization constant: q_z = Q / mean annual maximum.")
k[2].metric("Signal clarity S4", f"{res.s4:.2f}" if res.s4 is not None else "n/a",
            help="Median 4-week rolling std / global std of q_z. Values ≥ 0.5 indicate "
                 "noisy records where events may be poorly defined.")
k[3].metric("Hourly completeness", f"{res.completeness:.0%}")
for w in res.warnings:
    st.warning(w)

if ev_tab.empty:
    st.plotly_chart(daily_figure(res, p, None, rain), width="stretch")
    st.info("No events met the selection criteria. Try lowering the event threshold "
            "or the minimum peakness.")
    st.stop()

# selected event: default to the largest one; a click on the timeline overrides it
sig = (run["site_no"], str(run["start"]), str(run["end"]), p, run["use_peaks"])
tl_key = f"timeline_{abs(hash(sig))}"
if ss.get("event_sig") != sig or "event_choice" not in ss:
    ss.event_sig = sig
    ss.event_choice = int(ev_tab.loc[ev_tab["peak_qz"].idxmax(), "event"])
    ss.last_click = None
tl_state = ss.get(tl_key)
points = tl_state.selection.points if tl_state and tl_state.selection else []
if points:
    clicked = int(points[0]["point_index"]) + 1
    if ss.last_click != clicked:
        ss.last_click = clicked
        ss.event_choice = clicked
sel_ev = ev_tab.iloc[ss.event_choice - 1]

st.subheader("Discharge record and detected events")
st.caption("Daily aggregation of the hourly normalized discharge (daily max) and baseflow "
           "(daily mean). Shaded spans are the detected events.")
st.plotly_chart(daily_figure(res, p, sel_ev, rain), width="stretch",
                config=dict(displaylogo=False))
st.caption("**Event selector** — click a bar to inspect an event "
           "(bar position = event span, height = normalized peak).")
st.plotly_chart(timeline_figure(res, ss.event_choice - 1), key=tl_key, on_select="rerun",
                selection_mode="points", width="stretch",
                config=dict(displayModeBar=False))


def _step(d: int):
    ss.event_choice = int(np.clip(ss.event_choice + d, 1, len(ev_tab)))


n1, n2, n3 = st.columns([1, 6, 1], vertical_alignment="bottom")
n1.button("◀ Previous", on_click=_step, args=(-1,), width="stretch",
          disabled=ss.event_choice <= 1)
n2.selectbox("Event", ev_tab["event"].tolist(), key="event_choice",
             format_func=lambda i: event_label(ev_tab.iloc[i - 1]),
             label_visibility="collapsed")
n3.button("Next ▶", on_click=_step, args=(1,), width="stretch",
          disabled=ss.event_choice >= len(ev_tab))

st.subheader(f"Event {sel_ev.event}")
m = st.columns(4)
m[0].metric("Start", f"{sel_ev.start:%Y-%m-%d %H:%M}")
m[1].metric("Peak", f"{sel_ev.peak_time:%Y-%m-%d %H:%M}")
m[2].metric("Time to peak", fmt_hours(sel_ev.ttp_h))
m[3].metric("Duration", fmt_hours(sel_ev.duration_h))
m = st.columns(4)
m[0].metric("Peak discharge", f"{sel_ev.peak_cfs:,.0f} cfs",
            f"{sel_ev.peak_cfs * usgs.CFS_TO_CMS:,.1f} m³/s", delta_color="off")
m[1].metric("Peak / mean annual max", f"{sel_ev.peak_qz:.2f}")
m[2].metric("Peak stage",
            f"{sel_ev.peak_stage_ft:.2f} ft" if pd.notna(sel_ev.peak_stage_ft) else "n/a")
m[3].metric("Direct runoff depth",
            f"{sel_ev.runoff_depth_mm:.1f} mm" if pd.notna(sel_ev.runoff_depth_mm) else "n/a",
            f"{sel_ev.runoff_volume_m3 / 1e6:,.2f} hm³", delta_color="off",
            help="Separated runoff integrated over the event, divided by the upstream area.")
if rain is not None:
    m = st.columns(4)
    has = pd.notna(sel_ev.rain_mm)
    m[0].metric("Event rainfall", f"{sel_ev.rain_mm:.1f} mm" if has else "n/a",
                help=f"AORC basin average from {pl.RAIN_LEAD_HOURS} h before the event "
                     "start to the event end.")
    m[1].metric("Max intensity", f"{sel_ev.max_rain_mm_h:.1f} mm/h" if has else "n/a")
    m[2].metric("Runoff coefficient",
                f"{sel_ev.runoff_coef:.2f}" if pd.notna(sel_ev.runoff_coef) else "n/a",
                help="Direct runoff depth / event rainfall.")
    if not has:
        m[3].caption(f"No AORC rainfall for this event (AORC ends {aorc_end:%Y-%m-%d}).")
st.plotly_chart(event_figure(res, iv, sel_ev, rain), width="stretch",
                config=dict(displaylogo=False))
st.caption(f"Times are local standard time ({run['tz_cd']}, no daylight saving). "
           "Discharge and stage are the USGS instantaneous values; baseflow is the "
           "filter output on the hourly series, rescaled to cfs."
           + (" Rainfall is the hourly AORC basin average." if rain is not None else ""))

with st.expander(f"Event table ({len(ev_tab)} events)"):
    st.dataframe(
        ev_tab, hide_index=True, width="stretch",
        column_config={
            "start": st.column_config.DatetimeColumn("Start", format="YYYY-MM-DD HH:mm"),
            "end": st.column_config.DatetimeColumn("End", format="YYYY-MM-DD HH:mm"),
            "peak_time": st.column_config.DatetimeColumn("Peak", format="YYYY-MM-DD HH:mm"),
            "duration_h": st.column_config.NumberColumn("Duration (h)", format="%.0f"),
            "ttp_h": st.column_config.NumberColumn("Time to peak (h)", format="%.0f"),
            "peak_qz": st.column_config.NumberColumn("Peak (–)", format="%.3f"),
            "peak_cfs": st.column_config.NumberColumn("Peak (cfs)", format="localized"),
            "peak_stage_ft": st.column_config.NumberColumn("Peak stage (ft)", format="%.2f"),
            "missing_frac": st.column_config.NumberColumn("Missing", format="percent"),
            "runoff_volume_m3": st.column_config.NumberColumn("Runoff vol. (m³)",
                                                              format="localized"),
            "runoff_depth_mm": st.column_config.NumberColumn("Runoff depth (mm)",
                                                             format="%.1f"),
            "rain_mm": st.column_config.NumberColumn("Rainfall (mm)", format="%.1f"),
            "max_rain_mm_h": st.column_config.NumberColumn("Max rain (mm/h)",
                                                           format="%.1f"),
            "runoff_coef": st.column_config.NumberColumn("Runoff coef.", format="%.2f"),
        })

st.subheader("Data")
st.caption(f"Saved for this gauge in `{storage.gauge_dir(run['site_no'])}`. "
           "Files are read only when you click a button.")
d = st.columns(3)
for col, label, path in ((d[0], "USGS instantaneous data", iv_file),
                         (d[1], "AORC basin rainfall", rain_file),
                         (d[2], "Detected events", ev_file)):
    available = path.exists() and (path != rain_file or rain is not None)
    col.download_button(f"⬇ {label} (CSV)", data=lambda path=path: path.read_bytes(),
                        file_name=path.name, mime="text/csv", width="stretch",
                        disabled=not available,
                        help=None if available else "Enable AORC rainfall and run again.")
