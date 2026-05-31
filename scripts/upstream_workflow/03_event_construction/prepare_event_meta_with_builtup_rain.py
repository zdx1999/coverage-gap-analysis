#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DEFAULT_DATASET_DIR = Path(r"D:/SM/zdxrun/datatest/research/数据集")
DEFAULT_REFINED = Path(r"D:/SM/zdxrun/datatest/research/outputs_spatial_burden/refined_outputs")
DEFAULT_RAIN_CSV = Path(r"D:/SM/zdxrun/datatest/research/数据集/All_Events_Rainfall_TimeSeries_BuiltUp.csv")

def find_first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None

def infer_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not infer column from {candidates}. Available: {list(df.columns)}")
    return None

def infer_rain_col(df):
    preferred = ["rain_mm","rain","rainfall","precip","precip_mm","peak_rain","rain_value","grid_rain","grid_rain_mm","builtup_rain","rain_intensity","Rain","Rainfall"]
    for c in preferred:
        if c in df.columns:
            return c
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        cl = str(c).lower()
        if "rain" in cl or "precip" in cl:
            return c
    raise ValueError(f"Could not infer rain column. Available columns: {list(df.columns)}")

def load_base_event_meta(refined_dir: Path) -> pd.DataFrame:
    candidates = [
        refined_dir / "event_meta_screened.csv",
        refined_dir / "event_meta_screened_with_peak.csv",
        refined_dir / "event_meta_with_extreme_flags.csv",
        refined_dir / "event_meta_with_extreme_flags_region.csv",
    ]
    fp = find_first_existing(candidates)
    if fp is None:
        raise FileNotFoundError("Could not find a usable base event meta file.")
    print(f"Using event meta base: {fp}")
    return pd.read_csv(fp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--refined-dir", type=str, default=str(DEFAULT_REFINED))
    parser.add_argument("--rain-csv", type=str, default=None)
    parser.add_argument("--event-meta", type=str, default=None)
    parser.add_argument("--city-q", type=float, default=0.75)
    parser.add_argument("--national-q", type=float, default=0.75)
    parser.add_argument("--min-city-events", type=int, default=5)
    parser.add_argument("--peak-col-name", type=str, default="peak_rain_for_screen")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    refined_dir = Path(args.refined_dir)
    rain_csv = Path(args.rain_csv) if args.rain_csv else DEFAULT_RAIN_CSV
    if not rain_csv.exists():
        alt = dataset_dir / "All_Events_Rainfall_TimeSeries_BuiltUp.csv"
        if alt.exists():
            rain_csv = alt
        else:
            raise FileNotFoundError(f"Rainfall file not found: {rain_csv}")

    event_meta = pd.read_csv(Path(args.event_meta)) if args.event_meta else load_base_event_meta(refined_dir)
    rain = pd.read_csv(rain_csv)
    rain.columns = [str(c).strip() for c in rain.columns]
    print(f"Using rainfall file: {rain_csv}")

    event_col = infer_col(rain, ["Event_ID", "event_id", "EVENT_ID"])
    city_col = infer_col(rain, ["city_clean", "city", "City", "CITY"], required=False)
    rain_col = infer_rain_col(rain)

    rain[event_col] = pd.to_numeric(rain[event_col], errors="coerce")
    rain[rain_col] = pd.to_numeric(rain[rain_col], errors="coerce")
    rain = rain.dropna(subset=[event_col, rain_col]).copy()

    group_cols = [event_col] + ([city_col] if city_col is not None else [])
    peak = rain.groupby(group_cols, as_index=False)[rain_col].max().rename(columns={event_col: "Event_ID", rain_col: args.peak_col_name})
    if city_col is not None:
        peak = peak.rename(columns={city_col: "city_clean"})
    out_peak = refined_dir / "event_peak_rain_builtup.csv"
    peak.to_csv(out_peak, index=False, encoding="utf-8-sig")

    merge_keys = ["Event_ID"] + (["city_clean"] if "city_clean" in event_meta.columns and "city_clean" in peak.columns else [])
    meta = event_meta.copy()
    meta["Event_ID"] = pd.to_numeric(meta["Event_ID"], errors="coerce")
    peak["Event_ID"] = pd.to_numeric(peak["Event_ID"], errors="coerce")
    meta = meta.merge(peak, on=merge_keys, how="left")
    meta["Event_Peak_Rain"] = meta[args.peak_col_name]
    out_meta_peak = refined_dir / "event_meta_screened_with_peak_builtup.csv"
    meta.to_csv(out_meta_peak, index=False, encoding="utf-8-sig")

    work = meta.copy()
    if "city_clean" not in work.columns:
        raise ValueError("event meta must contain city_clean to compute city-level extreme flags.")
    work[args.peak_col_name] = pd.to_numeric(work[args.peak_col_name], errors="coerce")
    national_thr = work[args.peak_col_name].dropna().quantile(args.national_q)
    work["extreme_scope"] = "national_fallback"
    work["extreme_threshold"] = national_thr
    city_counts = work.groupby("city_clean")[args.peak_col_name].apply(lambda s: s.notna().sum()).to_dict()
    city_thr = {}
    for city, sub in work.groupby("city_clean"):
        vals = sub[args.peak_col_name].dropna()
        if city_counts.get(city, 0) >= args.min_city_events:
            city_thr[city] = vals.quantile(args.city_q)
        else:
            city_thr[city] = np.nan
    for city, thr in city_thr.items():
        if not pd.isna(thr):
            m = work["city_clean"] == city
            work.loc[m, "extreme_scope"] = "city_quantile"
            work.loc[m, "extreme_threshold"] = thr
    work["is_extreme"] = (pd.to_numeric(work[args.peak_col_name], errors="coerce") >= pd.to_numeric(work["extreme_threshold"], errors="coerce")).fillna(False).astype(int)

    out_meta_extreme = refined_dir / "event_meta_with_extreme_flags_builtup.csv"
    work.to_csv(out_meta_extreme, index=False, encoding="utf-8-sig")
    print(f"Saved peak-by-event: {out_peak}")
    print(f"Saved event meta with peak: {out_meta_peak}")
    print(f"Saved event meta with extreme flags: {out_meta_extreme}")
    print(f"Events with peak rain: {work[args.peak_col_name].notna().sum()}")
    print(f"Extreme events: {int(work['is_extreme'].sum())}")

if __name__ == "__main__":
    main()
