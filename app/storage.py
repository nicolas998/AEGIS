"""
Per-gauge data folder: data/USGS-<site_no>/ holds the downloaded USGS
instantaneous values, the AORC basin rainfall, the basin polygon and the
detected events. Files double as a cache that survives app restarts.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def gauge_dir(site_no: str) -> Path:
    d = DATA_DIR / f"USGS-{site_no}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tag(start: date, end: date) -> str:
    return f"{start:%Y%m%d}_{end:%Y%m%d}"


def iv_path(site_no: str, start: date, end: date) -> Path:
    return gauge_dir(site_no) / f"usgs_iv_{_tag(start, end)}.csv"


def rain_path(site_no: str, start: date, end: date) -> Path:
    return gauge_dir(site_no) / f"aorc_rain_{_tag(start, end)}.csv"


def events_path(site_no: str, start: date, end: date) -> Path:
    return gauge_dir(site_no) / f"events_{_tag(start, end)}.csv"


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=[0])


def write_table(df: pd.DataFrame | pd.Series, path: Path, index_label: str) -> None:
    df.to_csv(path, index_label=index_label)


def write_events(events: pd.DataFrame, path: Path, run_info: dict) -> None:
    """Events table plus the settings that produced it (same name, .json)."""
    events.to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(run_info, indent=2, default=str))


def write_basin(site_no: str, basin: dict | None) -> None:
    path = gauge_dir(site_no) / "basin.geojson"
    if basin and not path.exists():
        path.write_text(json.dumps(basin))
