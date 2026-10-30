"""
AEGIS workflow applied to a single gauge, built on the modules in ../code:

1. Preprocessing  (preproc.procs): hourly max resampling + normalization by
   the mean annual maximum, q_z = q(t) / mean(annual max).
2. Signal clarity (metric_calc.norm_roll_metric): S4 = median(rolling 4-week
   std) / global std. S4 >= ~0.5 flags records where extraction is unreliable.
3. Event detection (EventDetector): digital-filter baseflow separation,
   runoff thresholding, binarization and event screening. Optionally the
   runoff windows are split at troughs between peaks (event_split.py) so
   that every independent peak above the threshold becomes an event.
4. Optional clean-up: keep events with time-to-peak below a percentile.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ../code is not a package (and "code" shadows a stdlib module), so put it on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from EventDetector import EventDetector  # noqa: E402
from event_split import split_events  # noqa: E402
from metric_calc import norm_roll_metric  # noqa: E402
from preproc import procs  # noqa: E402

from usgs import CFS_TO_CMS  # noqa: E402

FILTERS = {
    "Nathan & McMahon (1990)": "Nathan",
    "Eckhardt (2005)": "Eckhardt",
    "Chapman & Maxwell (1996)": "Chapman",
}
SEPARATIONS = {
    "Peak-based split": "peaks",
    "AEGIS original (start to start)": "aegis",
}
S4_WINDOW_HOURS = 4 * 7 * 24
S4_LIMIT = 0.5
MIN_COMPLETENESS = 0.7


@dataclass(frozen=True)
class Params:
    filter_name: str = "Nathan & McMahon (1990)"
    fparam: float = 0.995
    bfi: float = 0.8
    event_threshold: float = 0.2
    runoff_threshold: float = 0.01
    max_nans: float = 0.2
    min_peakness: float = 0.6
    ttp_percentile: float | None = None
    separation: str = "aegis"     # "aegis" or "peaks" (code/event_split.py)
    min_rise: float = 0.1         # peaks mode: rise above the separating trough
    smooth_hours: int = 0         # peaks mode: moving average to locate peaks


@dataclass
class Result:
    hourly: pd.DataFrame          # Q_cfs, stage_ft, qz, baseflow, runoff
    events: pd.DataFrame          # one row per event
    qmax_ann: float               # mean annual maximum (cfs)
    n_years: int                  # years used for qmax_ann
    completeness: float           # fraction of non-missing hourly values
    s4: float | None
    n_detected: int               # events before the time-to-peak clean-up
    warnings: list[str] = field(default_factory=list)


def run_aegis(iv: pd.DataFrame, params: Params, area_km2: float | None = None,
              annual_peaks: pd.Series | None = None) -> Result:
    """
    Run the AEGIS workflow on instantaneous data (index: datetime,
    columns Q_cfs and stage_ft). If `annual_peaks` is given the mean of the
    USGS annual peak record is used for normalization instead of the mean
    annual maximum of the selected period.
    """
    warnings = []
    hourly = iv[["Q_cfs", "stage_ft"]].resample("1h").max()

    # 1. preprocessing (hourly max + normalization)
    if annual_peaks is not None and len(annual_peaks):
        qmax_ann = float(annual_peaks.mean())
        n_years = len(annual_peaks)
        hourly["qz"] = hourly["Q_cfs"] / qmax_ann
    else:
        hourly["qz"] = procs(iv, "Q_cfs")
        annual_max = hourly["Q_cfs"].resample("YE").max().dropna()
        qmax_ann = float(annual_max.mean())
        n_years = len(annual_max)
        if n_years < 3:
            warnings.append(
                f"The mean annual maximum is computed from only {n_years} calendar "
                "year(s). Use a longer period or the USGS annual-peak record for a "
                "more meaningful normalization.")

    completeness = float(hourly["Q_cfs"].notna().mean())
    if completeness < MIN_COMPLETENESS:
        warnings.append(
            f"Only {completeness:.0%} of the hourly record is complete "
            f"(AEGIS keeps gauges with > {MIN_COMPLETENESS:.0%}).")

    # 2. signal clarity
    s4 = None
    if hourly["qz"].notna().sum() > 2 * S4_WINDOW_HOURS:
        s4 = float(norm_roll_metric(hourly, column="qz", stat="std", agg="median",
                                    window_hours=S4_WINDOW_HOURS))
        if np.isnan(s4):
            s4 = None
        elif s4 >= S4_LIMIT:
            warnings.append(
                f"S4 = {s4:.2f} ≥ {S4_LIMIT}: the record is noisy and extracted "
                "events may be poorly defined; consider tuning the parameters.")

    # 3. event detection
    detector = EventDetector(
        runoff_threshold=params.runoff_threshold,
        event_threshold=params.event_threshold,
        max_nans=params.max_nans,
        min_peakness=params.min_peakness,
        filter_type=FILTERS[params.filter_name],
        fparam=params.fparam,
        bfi=params.bfi,
    )
    events, filtered, runoff_mask = detector.detect_events(hourly[["qz"]], only_events=False)
    hourly["baseflow"] = filtered["Baseflow"]
    hourly["runoff"] = filtered["Runoff"]
    qz = hourly["qz"].interpolate(limit_direction="both")

    if params.separation == "peaks":
        spans = split_events(qz, runoff_mask, hourly["qz"].isna().values,
                             event_threshold=params.event_threshold,
                             min_rise=params.min_rise, min_peakness=params.min_peakness,
                             max_nans=params.max_nans, smooth_hours=params.smooth_hours)
    else:
        spans = [(ev.index[0], ev.index[-1], ev["qz"].idxmax()) for ev in events]
    table = _event_table(spans, hourly, qz, qmax_ann, area_km2)
    n_detected = len(table)

    # 4. time-to-peak clean-up
    if params.ttp_percentile is not None and n_detected:
        limit = np.percentile(table["ttp_h"], params.ttp_percentile)
        table = table[table["ttp_h"] <= limit].reset_index(drop=True)
        table["event"] = np.arange(1, len(table) + 1)

    return Result(hourly, table, qmax_ann, n_years, completeness, s4,
                  n_detected, warnings)


RAIN_LEAD_HOURS = 12


def add_rainfall(events: pd.DataFrame, rain: pd.Series | None) -> pd.DataFrame:
    """
    Event rainfall from the hourly basin-average series: total and peak
    intensity from RAIN_LEAD_HOURS before the event start to its end, and the
    runoff coefficient (direct runoff depth / rainfall). NaN where the
    rainfall record does not cover the whole window.
    """
    out = events.copy()
    out["rain_mm"] = out["max_rain_mm_h"] = out["runoff_coef"] = np.nan
    if rain is None or rain.empty:
        return out
    lead = pd.Timedelta(hours=RAIN_LEAD_HOURS)
    for i, ev in out.iterrows():
        a, b = ev["start"] - lead, ev["end"]
        if a < rain.index[0] or b > rain.index[-1]:
            continue
        win = rain.loc[a:b]
        out.loc[i, "rain_mm"] = win.sum()
        out.loc[i, "max_rain_mm_h"] = win.max()
        if win.sum() > 0:
            out.loc[i, "runoff_coef"] = ev["runoff_depth_mm"] / win.sum()
    return out


def _event_table(spans: list[tuple], hourly: pd.DataFrame, qz: pd.Series,
                 qmax_ann: float, area_km2: float | None) -> pd.DataFrame:
    rows = []
    for start, end, peak_time in spans:
        win = hourly.loc[start:end]
        # direct runoff volume from the separated runoff, back in m3
        vol_m3 = float(win["runoff"].clip(lower=0).sum() * qmax_ann * CFS_TO_CMS * 3600)
        rows.append(dict(
            start=start, end=end, peak_time=peak_time,
            duration_h=(end - start) / pd.Timedelta(hours=1),
            ttp_h=(peak_time - start) / pd.Timedelta(hours=1),
            peak_qz=float(qz[peak_time]),
            peak_cfs=float(win["Q_cfs"].max()),
            peak_stage_ft=float(win["stage_ft"].max()),
            missing_frac=float(win["qz"].isna().mean()),
            runoff_volume_m3=vol_m3,
            runoff_depth_mm=vol_m3 / (area_km2 * 1e6) * 1e3 if area_km2 else np.nan,
        ))
    cols = ["start", "end", "peak_time", "duration_h", "ttp_h", "peak_qz",
            "peak_cfs", "peak_stage_ft", "missing_frac", "runoff_volume_m3",
            "runoff_depth_mm"]
    table = pd.DataFrame(rows, columns=cols)
    table.insert(0, "event", np.arange(1, len(table) + 1))
    return table
