#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define extreme events and recompute new_hotspot using city-specific baselines.

Inputs (default base dir):
  D:/SM/zdxrun/datatest/research/outputs_spatial_burden
Required files:
  - refined_outputs/city_event_grid_refined.csv
  - refined_outputs/event_concentration_metrics.csv
  - refined_outputs/event_meta_screened_with_peak.csv

Outputs:
  - refined_outputs/event_meta_with_extreme_flags.csv
  - refined_outputs/city_event_grid_with_new_hotspots.csv
  - refined_outputs/new_hotspot_summary_by_event.csv
  - refined_outputs/new_hotspot_summary_by_city.csv
  - refined_outputs/new_hotspot_definition_report.txt

Logic:
  1) Define extreme events from Event_Peak_Rain.
  2) Prefer city-specific q75 threshold when a city has enough events (default >= 8).
  3) Fall back to global q75 otherwise.
  4) new_hotspot = hotspot_refined == 1 in an extreme event AND the same grid was never a hotspot
     in any non-extreme event of the same city.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np

DEFAULT_BASE = Path(r"D:/SM/zdxrun/datatest/research/outputs_spatial_burden")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default=str(DEFAULT_BASE))
    p.add_argument("--city-min-events", type=int, default=8,
                   help="Minimum number of events in a city to use city-specific q75 threshold.")
    p.add_argument("--q", type=float, default=0.75,
                   help="Quantile for defining extreme events.")
    p.add_argument("--peak-col", type=str, default="Event_Peak_Rain")
    return p.parse_args()


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file for {name}: {path}")
    return pd.read_csv(path)


def infer_hotspot_col(df: pd.DataFrame) -> str:
    for c in ["hotspot_refined", "hotspot", "is_hotspot"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer hotspot column. Available columns: {list(df.columns)}")


def infer_grid_col(df: pd.DataFrame) -> str:
    for c in ["grid_id", "grid_key", "grid", "hex_id"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer grid id column. Available columns: {list(df.columns)}")


def build_extreme_flags(event_meta: pd.DataFrame, peak_col: str, city_min_events: int, q: float) -> pd.DataFrame:
    if peak_col not in event_meta.columns:
        raise ValueError(f"Peak column '{peak_col}' not found. Available columns: {list(event_meta.columns)}")

    out = event_meta.copy()
    out[peak_col] = pd.to_numeric(out[peak_col], errors="coerce")
    if out[peak_col].notna().sum() == 0:
        raise ValueError(f"Peak column '{peak_col}' contains no numeric values.")

    global_q = float(out[peak_col].quantile(q))

    city_stats = (
        out.groupby("city_clean", dropna=False)[peak_col]
           .agg(city_event_n="count", city_q=lambda s: s.quantile(q))
           .reset_index()
    )
    out = out.merge(city_stats, on="city_clean", how="left")

    out["extreme_threshold"] = np.where(
        out["city_event_n"] >= city_min_events,
        out["city_q"],
        global_q,
    )
    out["extreme_source"] = np.where(out["city_event_n"] >= city_min_events, "city_q", "global_q")
    out["is_extreme"] = (out[peak_col] >= out["extreme_threshold"]).astype(int)
    out["global_q_threshold"] = global_q
    return out


def compute_new_hotspots(grid_df: pd.DataFrame, event_meta_flags: pd.DataFrame, hotspot_col: str, grid_col: str) -> pd.DataFrame:
    cols = ["Event_ID", "city_clean", "is_extreme", "extreme_threshold", "extreme_source"]
    merged = grid_df.merge(event_meta_flags[cols], on=["Event_ID", "city_clean"], how="left", validate="many_to_one")

    merged[hotspot_col] = pd.to_numeric(merged[hotspot_col], errors="coerce").fillna(0).astype(int)
    merged["is_extreme"] = pd.to_numeric(merged["is_extreme"], errors="coerce").fillna(0).astype(int)

    non_extreme_hotspot = (
        merged.loc[(merged["is_extreme"] == 0) & (merged[hotspot_col] == 1), ["city_clean", grid_col]]
              .drop_duplicates()
              .assign(seen_non_extreme_hotspot=1)
    )

    merged = merged.merge(non_extreme_hotspot, on=["city_clean", grid_col], how="left")
    merged["seen_non_extreme_hotspot"] = merged["seen_non_extreme_hotspot"].fillna(0).astype(int)

    merged["new_hotspot_refined"] = (
        (merged["is_extreme"] == 1)
        & (merged[hotspot_col] == 1)
        & (merged["seen_non_extreme_hotspot"] == 0)
    ).astype(int)

    return merged


def summarize_by_event(df: pd.DataFrame, hotspot_col: str) -> pd.DataFrame:
    return (
        df.groupby(["Event_ID", "city_clean", "is_extreme"], dropna=False)
          .agg(
              grids=("Event_ID", "size"),
              hotspot_grids=(hotspot_col, "sum"),
              new_hotspot_grids=("new_hotspot_refined", "sum"),
              new_hotspot_share_within_hotspots=("new_hotspot_refined", lambda s: float(s.sum()) / np.nan if False else 0.0),
          )
          .reset_index()
    )


def summarize_by_event_better(df: pd.DataFrame, hotspot_col: str) -> pd.DataFrame:
    g = df.groupby(["Event_ID", "city_clean", "is_extreme"], dropna=False)
    out = g.agg(
        grids=("Event_ID", "size"),
        hotspot_grids=(hotspot_col, "sum"),
        new_hotspot_grids=("new_hotspot_refined", "sum"),
    ).reset_index()
    out["new_hotspot_share_within_hotspots"] = np.where(
        out["hotspot_grids"] > 0,
        out["new_hotspot_grids"] / out["hotspot_grids"],
        0.0,
    )
    return out


def summarize_by_city(df: pd.DataFrame, hotspot_col: str) -> pd.DataFrame:
    g = df.groupby("city_clean", dropna=False)
    out = g.agg(
        rows=("Event_ID", "size"),
        events=("Event_ID", "nunique"),
        hotspot_grids=(hotspot_col, "sum"),
        new_hotspot_grids=("new_hotspot_refined", "sum"),
        extreme_events=("is_extreme", "sum"),
    ).reset_index()
    out["new_hotspot_share_within_hotspots"] = np.where(
        out["hotspot_grids"] > 0,
        out["new_hotspot_grids"] / out["hotspot_grids"],
        0.0,
    )
    return out.sort_values(["new_hotspot_grids", "new_hotspot_share_within_hotspots"], ascending=[False, False])


def write_report(path: Path, event_meta_flags: pd.DataFrame, grid_with_new: pd.DataFrame, hotspot_col: str) -> None:
    total_events = int(event_meta_flags["Event_ID"].nunique())
    extreme_events = int(event_meta_flags.loc[event_meta_flags["is_extreme"] == 1, "Event_ID"].nunique())
    total_hotspots = int(grid_with_new[hotspot_col].sum())
    total_new_hotspots = int(grid_with_new["new_hotspot_refined"].sum())
    events_with_new = int(grid_with_new.loc[grid_with_new["new_hotspot_refined"] == 1, "Event_ID"].nunique())
    global_q = float(event_meta_flags["global_q_threshold"].iloc[0])
    city_threshold_events = int((event_meta_flags["extreme_source"] == "city_q").sum())
    global_threshold_events = int((event_meta_flags["extreme_source"] == "global_q").sum())

    text = f"""New-hotspot definition report
=============================
Total events: {total_events}
Extreme events: {extreme_events}
Global q-threshold used: {global_q:.4f}
Events using city-specific thresholds: {city_threshold_events}
Events using global threshold: {global_threshold_events}
Hotspot column used: {hotspot_col}
Total hotspot grids: {total_hotspots}
Total new-hotspot grids: {total_new_hotspots}
Share of hotspot grids that are new: {0 if total_hotspots == 0 else total_new_hotspots / total_hotspots:.4f}
Events with any new hotspot: {events_with_new}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    refined = base / "refined_outputs"

    grid_path = refined / "city_event_grid_refined.csv"
    metrics_path = refined / "event_concentration_metrics.csv"
    event_meta_path = refined / "event_meta_screened_with_peak.csv"

    grid_df = load_csv(grid_path, "city_event_grid_refined")
    _ = load_csv(metrics_path, "event_concentration_metrics")  # existence check only
    event_meta = load_csv(event_meta_path, "event_meta_screened_with_peak")

    hotspot_col = infer_hotspot_col(grid_df)
    grid_col = infer_grid_col(grid_df)

    event_meta_flags = build_extreme_flags(event_meta, args.peak_col, args.city_min_events, args.q)
    grid_with_new = compute_new_hotspots(grid_df, event_meta_flags, hotspot_col, grid_col)

    by_event = summarize_by_event_better(grid_with_new, hotspot_col)
    by_city = summarize_by_city(grid_with_new, hotspot_col)

    out_event_meta = refined / "event_meta_with_extreme_flags.csv"
    out_grid = refined / "city_event_grid_with_new_hotspots.csv"
    out_event = refined / "new_hotspot_summary_by_event.csv"
    out_city = refined / "new_hotspot_summary_by_city.csv"
    out_report = refined / "new_hotspot_definition_report.txt"

    event_meta_flags.to_csv(out_event_meta, index=False, encoding="utf-8-sig")
    grid_with_new.to_csv(out_grid, index=False, encoding="utf-8-sig")
    by_event.to_csv(out_event, index=False, encoding="utf-8-sig")
    by_city.to_csv(out_city, index=False, encoding="utf-8-sig")
    write_report(out_report, event_meta_flags, grid_with_new, hotspot_col)

    print("Done.")
    print(f"Wrote: {out_event_meta}")
    print(f"Wrote: {out_grid}")
    print(f"Wrote: {out_event}")
    print(f"Wrote: {out_city}")
    print(f"Wrote: {out_report}")


if __name__ == "__main__":
    main()
