#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Historical hotspot inventory coverage-gap analysis.

This script rebuilds the core objects for a "coverage gap under extreme
rainfall" storyline from the existing grid-event table.

Interpretation boundary:
    flood_count is a public-reporting / socially sensed impact measure. The
    event-time footprint here is a reported flood-impact footprint, not a
    hydrodynamic inundation boundary.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_FULL_TABLE = (
    REPO_ROOT
    / "data"
    / "city_event_grid_full_gaia_v4_poi_repaired_253cities.csv"
)
DEFAULT_ROAD_BASE = (
    REPO_ROOT
    / "data"
    / "city_grid_base_centroids_gaia_with_gee_terrain_repaired_253cities.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "coverage_gap_analysis"

FULL_USECOLS = [
    "Event_ID",
    "city_clean",
    "grid_id",
    "centroid_lon",
    "centroid_lat",
    "cell_size_m",
    "flood_count",
    "is_extreme",
    "hotspot_refined",
    "new_hotspot_region",
    "worldpop_2020",
    "poi_count_2018",
    "Event_Peak_Rain",
]

NUMERIC_COLS = [
    "Event_ID",
    "centroid_lon",
    "centroid_lat",
    "cell_size_m",
    "flood_count",
    "is_extreme",
    "hotspot_refined",
    "new_hotspot_region",
    "worldpop_2020",
    "poi_count_2018",
    "Event_Peak_Rain",
]

ROAD_L2_CANDIDATES = [
    "road_len_km_grip4_l2",
    "road_l2_len_km",
    "road_len_l2_km",
    "road_l2_km",
    "road_l2_length_km",
    "grip4_road_l2_len_km",
    "grip_l2_len_km",
    "l2_len_km",
]
ROAD_L3_CANDIDATES = [
    "road_len_km_grip4_l3",
    "road_l3_len_km",
    "road_len_l3_km",
    "road_l3_km",
    "road_l3_length_km",
    "grip4_road_l3_len_km",
    "grip_l3_len_km",
    "l3_len_km",
]

HOTSPOT_TOP_SHARES = [0.10, 0.20, 0.30]
IMPACT_DEFINITIONS = ["gt0", "ge2", "top50", "top75"]
PRIMARY_HOTSPOT_SHARE = 0.20
PRIMARY_IMPACT_DEF = "gt0"
PRIMARY_SCALE_M = 1000


def progress(msg: str) -> None:
    print(f"[coverage-gap] {msg}", flush=True)


def clean_city_name(value) -> str:
    if pd.isna(value):
        return ""
    out = str(value).strip().replace(" ", "")
    suffixes = [
        "\u7279\u522b\u884c\u653f\u533a",
        "\u81ea\u6cbb\u5dde",
        "\u5e02\u8f96\u533a",
        "\u5730\u533a",
        "\u81ea\u6cbb\u53bf",
        "\u5e02",
        "\u76df",
        "\u53bf",
        "\u533a",
    ]
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if out.endswith(suffix) and len(out) > len(suffix):
                out = out[: -len(suffix)]
                changed = True
                break
    return out


def first_existing(cols: Iterable[str], candidates: Iterable[str]) -> str | None:
    colset = set(cols)
    for c in candidates:
        if c in colset:
            return c
    lowered = {str(c).lower(): c for c in cols}
    for c in candidates:
        hit = lowered.get(c.lower())
        if hit is not None:
            return hit
    return None


def safe_div(numer: pd.Series | np.ndarray, denom: pd.Series | np.ndarray) -> np.ndarray:
    numer_arr = np.asarray(numer, dtype=float)
    denom_arr = np.asarray(denom, dtype=float)
    out = np.full_like(numer_arr, np.nan, dtype=float)
    mask = denom_arr > 0
    out[mask] = numer_arr[mask] / denom_arr[mask]
    return out


def iqr(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.quantile(0.75) - s.quantile(0.25))


def load_full_table(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = [c for c in FULL_USECOLS if c in header.columns]
    missing = [c for c in FULL_USECOLS if c not in header.columns]
    if missing:
        progress(f"missing optional/expected full-table columns: {missing}")
    required = [
        "Event_ID",
        "city_clean",
        "grid_id",
        "centroid_lon",
        "centroid_lat",
        "cell_size_m",
        "flood_count",
        "is_extreme",
        "worldpop_2020",
        "poi_count_2018",
        "Event_Peak_Rain",
    ]
    absent_required = [c for c in required if c not in available]
    if absent_required:
        raise KeyError(f"Missing required columns in full table: {absent_required}")

    progress(f"loading full grid-event table from {path}")
    df = pd.read_csv(path, usecols=available, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Event_ID"] = df["Event_ID"].fillna(-1).astype("int64")
    df = df[df["Event_ID"] >= 0].copy()
    df["grid_id"] = df["grid_id"].astype(str).str.strip()
    df["city_clean"] = df["city_clean"].astype(str).str.strip()
    df["city_std"] = df["city_clean"].map(clean_city_name)

    for c in ["flood_count", "is_extreme", "worldpop_2020", "poi_count_2018"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["flood_count"] = df["flood_count"].clip(lower=0)
    df["worldpop_2020"] = df["worldpop_2020"].clip(lower=0)
    df["poi_count_2018"] = df["poi_count_2018"].clip(lower=0)
    df["is_extreme"] = (df["is_extreme"] == 1).astype("int8")
    df["cell_size_m"] = pd.to_numeric(df["cell_size_m"], errors="coerce").fillna(1000)
    df["Event_Peak_Rain"] = pd.to_numeric(df["Event_Peak_Rain"], errors="coerce")
    return df


def infer_road_cols(path: Path) -> tuple[str | None, str | None]:
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    l2 = first_existing(cols, ROAD_L2_CANDIDATES)
    l3 = first_existing(cols, ROAD_L3_CANDIDATES)
    return l2, l3


def load_road_lookup(path: Path) -> tuple[pd.DataFrame, dict]:
    l2_col, l3_col = infer_road_cols(path)
    if l2_col is None and l3_col is None:
        progress("no secondary-road columns found; road exposure will be zero")
        empty = pd.DataFrame({"city_std": [], "grid_id": [], "secondary_road": []})
        return empty, {"road_l2_col": None, "road_l3_col": None, "road_rows": 0}

    usecols = ["city_clean", "grid_id"] + [c for c in [l2_col, l3_col] if c is not None]
    progress(f"loading road baseline from {path}")
    road = pd.read_csv(path, usecols=usecols, low_memory=False)
    road.columns = [str(c).strip() for c in road.columns]
    road["grid_id"] = road["grid_id"].astype(str).str.strip()
    road["city_std"] = road["city_clean"].map(clean_city_name)
    sec = np.zeros(len(road), dtype=float)
    for c in [l2_col, l3_col]:
        if c is not None:
            sec += pd.to_numeric(road[c], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    out = road[["city_std", "grid_id"]].copy()
    out["secondary_road"] = sec
    out = out.groupby(["city_std", "grid_id"], as_index=False)["secondary_road"].sum()
    meta = {
        "road_l2_col": l2_col,
        "road_l3_col": l3_col,
        "road_rows": len(out),
    }
    return out, meta


def attach_road(df: pd.DataFrame, road_lookup: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if road_lookup.empty:
        df["secondary_road"] = 0.0
        return df, {"road_merge_non_null_share": 0.0}
    progress("merging secondary-road exposure")
    merged = df.merge(road_lookup, on=["city_std", "grid_id"], how="left")
    non_null_share = float(merged["secondary_road"].notna().mean())
    merged["secondary_road"] = pd.to_numeric(merged["secondary_road"], errors="coerce").fillna(0)
    return merged, {"road_merge_non_null_share": non_null_share}


def lonlat_to_web_mercator(lon: pd.Series, lat: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    r = 6378137.0
    lon_arr = pd.to_numeric(lon, errors="coerce").to_numpy(dtype=float)
    lat_arr = pd.to_numeric(lat, errors="coerce").clip(-85.05112878, 85.05112878).to_numpy(dtype=float)
    x = r * np.deg2rad(lon_arr)
    y = r * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat_arr) / 2.0))
    return x, y


def prepare_scale_table(df: pd.DataFrame, scale_m: int) -> pd.DataFrame:
    if scale_m == PRIMARY_SCALE_M:
        out = df.copy()
        out["analysis_grid_id"] = out["grid_id"]
        out["analysis_scale_m"] = scale_m
        return out

    progress(f"aggregating grid-event table to {scale_m} m diagnostic scale")
    work = df.copy()
    x, y = lonlat_to_web_mercator(work["centroid_lon"], work["centroid_lat"])
    gx = np.floor(x / float(scale_m)).astype("int64")
    gy = np.floor(y / float(scale_m)).astype("int64")
    work["analysis_grid_id"] = pd.Series(gx, index=work.index).astype(str) + "_" + pd.Series(gy, index=work.index).astype(str)

    group_cols = ["city_std", "city_clean", "Event_ID", "analysis_grid_id"]
    agg = work.groupby(group_cols, as_index=False).agg(
        centroid_lon=("centroid_lon", "mean"),
        centroid_lat=("centroid_lat", "mean"),
        cell_size_m=("cell_size_m", "max"),
        flood_count=("flood_count", "sum"),
        is_extreme=("is_extreme", "max"),
        worldpop_2020=("worldpop_2020", "sum"),
        poi_count_2018=("poi_count_2018", "sum"),
        secondary_road=("secondary_road", "sum"),
        Event_Peak_Rain=("Event_Peak_Rain", "max"),
    )
    agg["cell_size_m"] = scale_m
    agg["analysis_scale_m"] = scale_m
    return agg


def assign_historical_inventory(ext: pd.DataFrame, top_share: float) -> pd.Series:
    positive = ext["hist_reports_loo"] > 0
    flags = pd.Series(False, index=ext.index)
    if not positive.any():
        return flags

    pos = ext.loc[positive, ["city_std", "Event_ID", "analysis_grid_id", "hist_reports_loo"]].copy()
    pos = pos.sort_values(
        ["city_std", "Event_ID", "hist_reports_loo", "analysis_grid_id"],
        ascending=[True, True, False, True],
    )
    group_cols = ["city_std", "Event_ID"]
    pos["_rank"] = pos.groupby(group_cols).cumcount() + 1
    pos["_n_positive_history"] = pos.groupby(group_cols)["hist_reports_loo"].transform("size")
    pos["_n_keep"] = np.ceil(pos["_n_positive_history"] * top_share).astype(int).clip(lower=1)
    keep_idx = pos.index[pos["_rank"] <= pos["_n_keep"]]
    flags.loc[keep_idx] = True
    return flags


def build_affected_flags(ext: pd.DataFrame, definition: str) -> pd.Series:
    flood = pd.to_numeric(ext["flood_count"], errors="coerce").fillna(0)
    if definition == "gt0":
        return flood > 0
    if definition == "ge2":
        return flood >= 2
    if definition in {"top50", "top75"}:
        q = 0.50 if definition == "top50" else 0.75
        positive = flood.where(flood > 0)
        threshold = positive.groupby([ext["city_std"], ext["Event_ID"]]).transform(
            lambda s: s.dropna().quantile(q) if s.notna().any() else np.nan
        )
        return (flood > 0) & (flood >= threshold)
    raise ValueError(f"Unknown event-impact definition: {definition}")


def summarize_variant(
    ext: pd.DataFrame,
    historical_hotspot: pd.Series,
    affected: pd.Series,
    top_share: float,
    impact_definition: str,
    scale_m: int,
) -> pd.DataFrame:
    work = ext[[
        "city_std",
        "city_clean",
        "Event_ID",
        "is_extreme",
        "Event_Peak_Rain",
        "flood_count",
        "worldpop_2020",
        "secondary_road",
        "poi_count_2018",
    ]].copy()
    hist = historical_hotspot.reindex(work.index).fillna(False).to_numpy(dtype=bool)
    aff = affected.reindex(work.index).fillna(False).to_numpy(dtype=bool)
    flood = pd.to_numeric(work["flood_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
    pop = pd.to_numeric(work["worldpop_2020"], errors="coerce").fillna(0).to_numpy(dtype=float)
    road = pd.to_numeric(work["secondary_road"], errors="coerce").fillna(0).to_numpy(dtype=float)
    poi = pd.to_numeric(work["poi_count_2018"], errors="coerce").fillna(0).to_numpy(dtype=float)

    covered = aff & hist
    uncovered = aff & (~hist)
    work["_historical_hotspot"] = hist.astype("int8")
    work["_affected"] = aff.astype("int8")
    work["_covered"] = covered.astype("int8")
    work["_uncovered"] = uncovered.astype("int8")
    work["_reports_total"] = np.where(aff, flood, 0.0)
    work["_reports_covered"] = np.where(covered, flood, 0.0)
    work["_reports_uncovered"] = np.where(uncovered, flood, 0.0)
    work["_pop_total"] = np.where(aff, pop, 0.0)
    work["_pop_uncovered"] = np.where(uncovered, pop, 0.0)
    work["_road_total"] = np.where(aff, road, 0.0)
    work["_road_uncovered"] = np.where(uncovered, road, 0.0)
    work["_func_total"] = np.where(aff, poi, 0.0)
    work["_func_uncovered"] = np.where(uncovered, poi, 0.0)

    grouped = work.groupby(["city_std", "city_clean", "Event_ID"], as_index=False).agg(
        is_extreme=("is_extreme", "max"),
        Event_Peak_Rain=("Event_Peak_Rain", "max"),
        total_affected_grids=("_affected", "sum"),
        total_reports=("_reports_total", "sum"),
        historical_hotspot_grids=("_historical_hotspot", "sum"),
        covered_affected_grids=("_covered", "sum"),
        uncovered_affected_grids=("_uncovered", "sum"),
        reports_covered=("_reports_covered", "sum"),
        reports_uncovered=("_reports_uncovered", "sum"),
        population_exposure_total=("_pop_total", "sum"),
        population_exposure_uncovered=("_pop_uncovered", "sum"),
        road_exposure_total=("_road_total", "sum"),
        road_exposure_uncovered=("_road_uncovered", "sum"),
        function_exposure_total=("_func_total", "sum"),
        function_exposure_uncovered=("_func_uncovered", "sum"),
    )
    grouped = grouped[grouped["total_affected_grids"] > 0].copy()
    grouped["impact_coverage_rate"] = safe_div(
        grouped["covered_affected_grids"], grouped["total_affected_grids"]
    )
    grouped["uncovered_impact_share"] = safe_div(
        grouped["uncovered_affected_grids"], grouped["total_affected_grids"]
    )
    grouped["report_weighted_coverage_rate"] = safe_div(
        grouped["reports_covered"], grouped["total_reports"]
    )
    grouped["report_weighted_uncovered_share"] = safe_div(
        grouped["reports_uncovered"], grouped["total_reports"]
    )
    grouped["population_exposure_outside_historical_hotspots"] = safe_div(
        grouped["population_exposure_uncovered"], grouped["population_exposure_total"]
    )
    grouped["road_exposure_outside_historical_hotspots"] = safe_div(
        grouped["road_exposure_uncovered"], grouped["road_exposure_total"]
    )
    grouped["function_exposure_outside_historical_hotspots"] = safe_div(
        grouped["function_exposure_uncovered"], grouped["function_exposure_total"]
    )
    grouped["historical_hotspot_definition"] = f"loo_positive_history_top_{int(top_share * 100)}pct"
    grouped["hotspot_top_share"] = top_share
    grouped["event_impact_definition"] = impact_definition
    grouped["analysis_scale_m"] = scale_m
    return grouped


def run_scale_analysis(df: pd.DataFrame, scale_m: int, keep_grid_for_map: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    scaled = prepare_scale_table(df, scale_m)
    progress(f"computing leave-one-event-out historical reports at {scale_m} m")
    grid_total = (
        scaled.groupby(["city_std", "analysis_grid_id"], observed=True)["flood_count"]
        .sum()
        .rename("city_grid_total_reports")
        .reset_index()
    )
    scaled = scaled.merge(grid_total, on=["city_std", "analysis_grid_id"], how="left")
    ext = scaled[scaled["is_extreme"] == 1].copy()
    ext["hist_reports_loo"] = (
        pd.to_numeric(ext["city_grid_total_reports"], errors="coerce").fillna(0)
        - pd.to_numeric(ext["flood_count"], errors="coerce").fillna(0)
    ).clip(lower=0)
    ext = ext.sort_values(["city_std", "Event_ID", "analysis_grid_id"]).copy()

    event_rows = []
    map_grid = None
    primary_history = None
    primary_affected = None
    for top_share in HOTSPOT_TOP_SHARES:
        progress(f"assigning historical inventory top {int(top_share * 100)}% at {scale_m} m")
        hist_flag = assign_historical_inventory(ext, top_share)
        if top_share == PRIMARY_HOTSPOT_SHARE:
            primary_history = hist_flag.copy()
        affected_cache = {}
        for impact_def in IMPACT_DEFINITIONS:
            if impact_def not in affected_cache:
                affected_cache[impact_def] = build_affected_flags(ext, impact_def)
            progress(f"summarizing top {int(top_share * 100)}%, impact={impact_def}, scale={scale_m} m")
            summary = summarize_variant(
                ext,
                hist_flag,
                affected_cache[impact_def],
                top_share,
                impact_def,
                scale_m,
            )
            event_rows.append(summary)
            if top_share == PRIMARY_HOTSPOT_SHARE and impact_def == PRIMARY_IMPACT_DEF:
                primary_affected = affected_cache[impact_def].copy()

    all_events = pd.concat(event_rows, ignore_index=True)
    if keep_grid_for_map and primary_history is not None and primary_affected is not None:
        map_grid = ext[[
            "city_std",
            "city_clean",
            "Event_ID",
            "analysis_grid_id",
            "centroid_lon",
            "centroid_lat",
            "cell_size_m",
            "flood_count",
            "Event_Peak_Rain",
        ]].copy()
        map_grid["historical_hotspot_inventory"] = primary_history.reindex(ext.index).fillna(False).to_numpy(dtype=bool)
        map_grid["event_time_reported_impact"] = primary_affected.reindex(ext.index).fillna(False).to_numpy(dtype=bool)
        map_grid["covered_event_impact"] = (
            map_grid["historical_hotspot_inventory"] & map_grid["event_time_reported_impact"]
        )
        map_grid["uncovered_event_impact"] = (
            (~map_grid["historical_hotspot_inventory"]) & map_grid["event_time_reported_impact"]
        )
    return all_events, map_grid


def make_primary_event_table(all_events: pd.DataFrame) -> pd.DataFrame:
    primary = all_events[
        (all_events["analysis_scale_m"] == PRIMARY_SCALE_M)
        & (np.isclose(all_events["hotspot_top_share"], PRIMARY_HOTSPOT_SHARE))
        & (all_events["event_impact_definition"] == PRIMARY_IMPACT_DEF)
    ].copy()
    cols = [
        "city_clean",
        "city_std",
        "Event_ID",
        "is_extreme",
        "Event_Peak_Rain",
        "total_affected_grids",
        "total_reports",
        "historical_hotspot_grids",
        "covered_affected_grids",
        "uncovered_affected_grids",
        "impact_coverage_rate",
        "uncovered_impact_share",
        "report_weighted_coverage_rate",
        "report_weighted_uncovered_share",
        "population_exposure_outside_historical_hotspots",
        "road_exposure_outside_historical_hotspots",
        "function_exposure_outside_historical_hotspots",
        "historical_hotspot_definition",
        "event_impact_definition",
        "analysis_scale_m",
    ]
    return primary[cols].sort_values(["city_std", "Event_ID"]).reset_index(drop=True)


def make_city_table(primary: pd.DataFrame) -> pd.DataFrame:
    def prop_gt(s: pd.Series, threshold: float) -> float:
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if vals.empty:
            return np.nan
        return float((vals > threshold).mean())

    grouped = primary.groupby(["city_std", "city_clean"], as_index=False).agg(
        number_of_extreme_events=("Event_ID", "nunique"),
        median_uncovered_impact_share=("uncovered_impact_share", "median"),
        mean_uncovered_impact_share=("uncovered_impact_share", "mean"),
        median_population_exposure_outside=("population_exposure_outside_historical_hotspots", "median"),
        median_road_exposure_outside=("road_exposure_outside_historical_hotspots", "median"),
        median_function_exposure_outside=("function_exposure_outside_historical_hotspots", "median"),
        proportion_events_uncovered_share_gt_25=("uncovered_impact_share", lambda s: prop_gt(s, 0.25)),
        proportion_events_uncovered_share_gt_50=("uncovered_impact_share", lambda s: prop_gt(s, 0.50)),
    )
    return grouped.sort_values(["median_uncovered_impact_share", "number_of_extreme_events"], ascending=[False, False])


def make_sensitivity_table(all_events: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "uncovered_impact_share",
        "report_weighted_uncovered_share",
        "population_exposure_outside_historical_hotspots",
        "road_exposure_outside_historical_hotspots",
        "function_exposure_outside_historical_hotspots",
    ]
    rows = []
    group_cols = [
        "analysis_scale_m",
        "historical_hotspot_definition",
        "hotspot_top_share",
        "event_impact_definition",
    ]
    for keys, sub in all_events.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["event_count"] = int(len(sub))
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"median_{col}"] = float(vals.median()) if vals.notna().any() else np.nan
            row[f"iqr_{col}"] = iqr(vals)
            row[f"q25_{col}"] = float(vals.quantile(0.25)) if vals.notna().any() else np.nan
            row[f"q75_{col}"] = float(vals.quantile(0.75)) if vals.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_audit(df: pd.DataFrame, road_meta: dict, merge_meta: dict) -> dict:
    audit = {}
    audit["rows_loaded"] = int(len(df))
    audit["cities"] = int(df["city_std"].nunique())
    audit["events"] = int(df["Event_ID"].nunique())
    audit["extreme_events"] = int(df.loc[df["is_extreme"] == 1, "Event_ID"].nunique())
    audit["cell_size_values"] = (
        df["cell_size_m"].round().value_counts(dropna=False).head(10).to_dict()
    )
    audit.update(road_meta)
    audit.update(merge_meta)
    city_grid_counts = df.groupby("city_std")["grid_id"].nunique().rename("city_grid_n")
    event_grid_counts = (
        df.groupby(["city_std", "Event_ID"])["grid_id"].nunique().rename("event_grid_n").reset_index()
    )
    event_grid_counts = event_grid_counts.merge(city_grid_counts.reset_index(), on="city_std", how="left")
    event_grid_counts["event_grid_to_city_grid_ratio"] = safe_div(
        event_grid_counts["event_grid_n"], event_grid_counts["city_grid_n"]
    )
    audit["event_grid_ratio_median"] = float(event_grid_counts["event_grid_to_city_grid_ratio"].median())
    audit["event_grid_ratio_p05"] = float(event_grid_counts["event_grid_to_city_grid_ratio"].quantile(0.05))
    audit["event_grid_ratio_min"] = float(event_grid_counts["event_grid_to_city_grid_ratio"].min())
    return audit


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    progress(f"wrote {path} ({len(df)} rows)")


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def plot_event_distribution(primary: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    metrics = [
        ("uncovered_impact_share", "Uncovered\nimpact"),
        ("population_exposure_outside_historical_hotspots", "Population\noutside"),
        ("road_exposure_outside_historical_hotspots", "Road\noutside"),
        ("function_exposure_outside_historical_hotspots", "Function\noutside"),
    ]
    vals = [primary[c].dropna().clip(0, 1).to_numpy() * 100 for c, _ in metrics]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bp = ax.boxplot(vals, patch_artist=True, showfliers=False, widths=0.48)
    colors = ["#F2A65A", "#6FB1A3", "#7A8CC7", "#C46AA7"]
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor=color, edgecolor="#333333", alpha=0.75)
    rng = np.random.default_rng(416)
    for i, arr in enumerate(vals, start=1):
        x = rng.normal(i, 0.055, size=len(arr))
        ax.scatter(x, arr, s=9, color="#222222", alpha=0.16, linewidths=0)
    ax.set_xticks(range(1, len(metrics) + 1), [label for _, label in metrics])
    ax.set_ylabel("Share of event-time reported impact/exposure (%)")
    ax.set_ylim(-3, 103)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_title("Extreme-event coverage gap distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    progress(f"wrote {out_path}")


def plot_coverage_vs_rainfall(primary: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    sub = primary[["Event_Peak_Rain", "uncovered_impact_share"]].dropna().copy()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(
        sub["Event_Peak_Rain"],
        sub["uncovered_impact_share"] * 100,
        s=18,
        color="#D36B35",
        edgecolors="white",
        linewidths=0.35,
        alpha=0.70,
    )
    if len(sub) >= 3 and sub["Event_Peak_Rain"].nunique() > 1:
        x = sub["Event_Peak_Rain"].to_numpy(dtype=float)
        y = (sub["uncovered_impact_share"] * 100).to_numpy(dtype=float)
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#222222", linewidth=1.1)
    r = sub["Event_Peak_Rain"].corr(sub["uncovered_impact_share"], method="pearson") if len(sub) > 2 else np.nan
    rho = sub["Event_Peak_Rain"].corr(sub["uncovered_impact_share"], method="spearman") if len(sub) > 2 else np.nan
    ax.text(
        0.03,
        0.97,
        f"Pearson r={r:.2f}\nSpearman rho={rho:.2f}\nn={len(sub)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="#BBBBBB", boxstyle="square,pad=0.25", alpha=0.9),
    )
    ax.set_xlabel("Event peak rainfall")
    ax.set_ylabel("Uncovered impact share (%)")
    ax.set_ylim(-3, 103)
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.set_title("Coverage gap vs. rainfall intensity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    progress(f"wrote {out_path}")


def plot_exposure_coupling(primary: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    xcol = "uncovered_impact_share"
    ycol = "function_exposure_outside_historical_hotspots"
    sub = primary[[xcol, ycol]].dropna().copy()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(
        sub[xcol] * 100,
        sub[ycol] * 100,
        s=18,
        color="#4B8F8C",
        edgecolors="white",
        linewidths=0.35,
        alpha=0.70,
    )
    if len(sub) >= 3:
        x = (sub[xcol] * 100).to_numpy(dtype=float)
        y = (sub[ycol] * 100).to_numpy(dtype=float)
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(0, 100, 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#222222", linewidth=1.1)
    r = sub[xcol].corr(sub[ycol], method="pearson") if len(sub) > 2 else np.nan
    rho = sub[xcol].corr(sub[ycol], method="spearman") if len(sub) > 2 else np.nan
    ax.text(
        0.03,
        0.97,
        f"Pearson r={r:.2f}\nSpearman rho={rho:.2f}\nn={len(sub)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="#BBBBBB", boxstyle="square,pad=0.25", alpha=0.9),
    )
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Uncovered impact share (%)")
    ax.set_ylabel("Function exposure outside historical hotspots (%)")
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.set_title("Spatial gap and functional exposure coupling")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    progress(f"wrote {out_path}")


def representative_specs() -> list[tuple[str, str, int]]:
    return [
        ("Zhengzhou", "\u90d1\u5dde", 5551),
        ("Guangzhou", "\u5e7f\u5dde", 4520),
        ("Beijing", "\u5317\u4eac", 7098),
        ("Wuhan", "\u6b66\u6c49", 3856),
        ("Chengdu", "\u6210\u90fd", 3215),
    ]


def plot_representative_maps(map_grid: pd.DataFrame, primary: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    specs = representative_specs()
    fig, axes = plt.subplots(1, len(specs), figsize=(14.5, 3.8))
    if len(specs) == 1:
        axes = [axes]
    for ax, (label, city_std, event_id) in zip(axes, specs):
        chosen_event = event_id
        sub = map_grid[(map_grid["city_std"] == city_std) & (map_grid["Event_ID"] == event_id)].copy()
        if sub.empty:
            candidates = primary[primary["city_std"] == city_std].sort_values("total_reports", ascending=False)
            if not candidates.empty:
                chosen_event = int(candidates.iloc[0]["Event_ID"])
                sub = map_grid[(map_grid["city_std"] == city_std) & (map_grid["Event_ID"] == chosen_event)].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{label}\nno event rows")
            continue

        show = sub[sub["historical_hotspot_inventory"] | sub["event_time_reported_impact"]].copy()
        if show.empty:
            show = sub.copy()
        lon = pd.to_numeric(show["centroid_lon"], errors="coerce")
        lat = pd.to_numeric(show["centroid_lat"], errors="coerce")
        lon_span = max(float(lon.max() - lon.min()), 0.02)
        lat_span = max(float(lat.max() - lat.min()), 0.02)
        ax.set_xlim(float(lon.min() - lon_span * 0.08), float(lon.max() + lon_span * 0.08))
        ax.set_ylim(float(lat.min() - lat_span * 0.08), float(lat.max() + lat_span * 0.08))

        hist = show[show["historical_hotspot_inventory"]]
        covered = show[show["covered_event_impact"]]
        uncovered = show[show["uncovered_event_impact"]]
        reports = show[pd.to_numeric(show["flood_count"], errors="coerce").fillna(0) > 0]

        ax.scatter(
            hist["centroid_lon"],
            hist["centroid_lat"],
            marker="s",
            s=18,
            facecolor="#E0E0E0",
            edgecolor="#9E9E9E",
            linewidth=0.25,
            alpha=0.65,
            label="Historical hotspot inventory",
            zorder=1,
        )
        ax.scatter(
            covered["centroid_lon"],
            covered["centroid_lat"],
            marker="s",
            s=26,
            facecolor="#2C7BB6",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.88,
            label="Covered event impact",
            zorder=2,
        )
        ax.scatter(
            uncovered["centroid_lon"],
            uncovered["centroid_lat"],
            marker="s",
            s=28,
            facecolor="#D7191C",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.88,
            label="Uncovered event impact",
            zorder=3,
        )
        if not reports.empty:
            sizes = 8 + 24 * np.sqrt(pd.to_numeric(reports["flood_count"], errors="coerce").fillna(0).clip(upper=50) / 50)
            ax.scatter(
                reports["centroid_lon"],
                reports["centroid_lat"],
                s=sizes,
                color="#111111",
                alpha=0.50,
                linewidth=0,
                label="Flood reports",
                zorder=4,
            )
        row = primary[(primary["city_std"] == city_std) & (primary["Event_ID"] == chosen_event)]
        if not row.empty:
            share = float(row.iloc[0]["uncovered_impact_share"]) * 100
            ax.set_title(f"{label}\nEvent {chosen_event}, gap {share:.0f}%", fontsize=9)
        else:
            ax.set_title(f"{label}\nEvent {chosen_event}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.45)
            spine.set_color("#BBBBBB")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Representative coverage-gap maps (reported impact footprint)", y=0.98, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    progress(f"wrote {out_path}")


def support_classification(primary: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    med_gap = float(primary["uncovered_impact_share"].median())
    prop_gt25 = float((primary["uncovered_impact_share"] > 0.25).mean())
    med_pop = float(primary["population_exposure_outside_historical_hotspots"].median())
    med_road = float(primary["road_exposure_outside_historical_hotspots"].median())
    med_func = float(primary["function_exposure_outside_historical_hotspots"].median())
    one_k = sensitivity[sensitivity["analysis_scale_m"] == PRIMARY_SCALE_M]
    sensitivity_ok = bool((one_k["median_uncovered_impact_share"] > 0.25).mean() >= 0.75) if not one_k.empty else False

    if (
        med_gap > 0.30
        and prop_gt25 > 0.60
        and np.nanmedian([med_pop, med_road, med_func]) > 0.30
        and sensitivity_ok
    ):
        return "Strong support"
    if med_gap > 0.15 or prop_gt25 > 0.40:
        return "Moderate support"
    return "Not supported"


def write_technical_note(
    out_path: Path,
    primary: pd.DataFrame,
    city: pd.DataFrame,
    sensitivity: pd.DataFrame,
    audit: dict,
) -> None:
    n_events = len(primary)
    n_cities = primary["city_std"].nunique()
    med_gap = primary["uncovered_impact_share"].median()
    mean_gap = primary["uncovered_impact_share"].mean()
    prop_gt25 = (primary["uncovered_impact_share"] > 0.25).mean()
    prop_gt50 = (primary["uncovered_impact_share"] > 0.50).mean()
    med_pop = primary["population_exposure_outside_historical_hotspots"].median()
    med_road = primary["road_exposure_outside_historical_hotspots"].median()
    med_func = primary["function_exposure_outside_historical_hotspots"].median()
    rain_r = primary["Event_Peak_Rain"].corr(primary["uncovered_impact_share"], method="pearson")
    func_r = primary["uncovered_impact_share"].corr(
        primary["function_exposure_outside_historical_hotspots"], method="pearson"
    )
    func_rho = primary["uncovered_impact_share"].corr(
        primary["function_exposure_outside_historical_hotspots"], method="spearman"
    )
    verdict = support_classification(primary, sensitivity)

    top_events = primary.sort_values(
        ["uncovered_impact_share", "total_affected_grids"], ascending=[False, False]
    ).head(8)
    weak_events = primary.sort_values(
        ["uncovered_impact_share", "total_affected_grids"], ascending=[True, False]
    ).head(8)
    strong_cities = city.sort_values(
        ["median_uncovered_impact_share", "number_of_extreme_events"], ascending=[False, False]
    ).head(8)

    sens_primary_scale = sensitivity[
        (sensitivity["analysis_scale_m"] == PRIMARY_SCALE_M)
        & (sensitivity["event_impact_definition"] == PRIMARY_IMPACT_DEF)
    ][
        [
            "historical_hotspot_definition",
            "event_count",
            "median_uncovered_impact_share",
            "iqr_uncovered_impact_share",
            "median_population_exposure_outside_historical_hotspots",
            "median_road_exposure_outside_historical_hotspots",
            "median_function_exposure_outside_historical_hotspots",
        ]
    ]

    lines = []
    lines.append("# Technical note: historical hotspot inventory coverage gap")
    lines.append("")
    lines.append("## Data support")
    lines.append(
        f"- Required fields are available in the current main table. Loaded {audit['rows_loaded']:,} grid-event rows, "
        f"{audit['cities']} cities, {audit['events']} events, and {audit['extreme_events']} extreme events."
    )
    lines.append(
        f"- Main analysis uses {n_events} city-extreme-event rows across {n_cities} cities after requiring a non-empty reported impact footprint."
    )
    lines.append(
        f"- The event-grid/full-city coverage audit has median event-grid/city-grid ratio "
        f"{audit['event_grid_ratio_median']:.3f}, p05 {audit['event_grid_ratio_p05']:.3f}, min {audit['event_grid_ratio_min']:.3f}."
    )
    lines.append(
        f"- Road exposure is L2+L3 from `{audit.get('road_l2_col')}` and `{audit.get('road_l3_col')}`; "
        f"road merge non-null share is {audit.get('road_merge_non_null_share', np.nan):.3f}."
    )
    lines.append("")
    lines.append("## Primary definition")
    lines.append(
        "- Historical hotspot inventory: within each city-event, leave the current Event_ID out, "
        "rank grids with positive historical reports by cumulative historical flood_count, and keep the top 20%."
    )
    lines.append(
        "- Event-time reported impact footprint: grids with flood_count > 0 in an extreme event."
    )
    lines.append("")
    lines.append("## Main results")
    lines.append(
        f"- Median uncovered impact share: {med_gap:.1%}; mean: {mean_gap:.1%}."
    )
    lines.append(
        f"- Events with uncovered impact share >25%: {prop_gt25:.1%}; >50%: {prop_gt50:.1%}."
    )
    lines.append(
        f"- Median exposure outside historical hotspots: population {med_pop:.1%}, road {med_road:.1%}, function/POI {med_func:.1%}."
    )
    lines.append(
        f"- Rainfall-gap association: Pearson r={rain_r:.2f}. Gap-function coupling: Pearson r={func_r:.2f}, Spearman rho={func_rho:.2f}."
    )
    lines.append(f"- Overall evidence classification: **{verdict}**.")
    lines.append("")
    lines.append("## Threshold robustness at the native 1 km grid")
    if sens_primary_scale.empty:
        lines.append("- No native-scale sensitivity rows were produced.")
    else:
        for _, row in sens_primary_scale.iterrows():
            lines.append(
                f"- {row['historical_hotspot_definition']}: median uncovered impact "
                f"{row['median_uncovered_impact_share']:.1%} "
                f"(IQR {row['iqr_uncovered_impact_share']:.1%}); "
                f"median outside exposure pop/road/function = "
                f"{row['median_population_exposure_outside_historical_hotspots']:.1%}/"
                f"{row['median_road_exposure_outside_historical_hotspots']:.1%}/"
                f"{row['median_function_exposure_outside_historical_hotspots']:.1%}."
            )
    lines.append("")
    lines.append("## Cities and events that most support the story")
    for _, row in strong_cities.iterrows():
        lines.append(
            f"- {row['city_clean']} ({row['number_of_extreme_events']} events): "
            f"median uncovered impact {row['median_uncovered_impact_share']:.1%}, "
            f"median function outside {row['median_function_exposure_outside']:.1%}."
        )
    lines.append("")
    lines.append("## Highest-gap events")
    for _, row in top_events.iterrows():
        lines.append(
            f"- {row['city_clean']} Event {int(row['Event_ID'])}: "
            f"uncovered impact {row['uncovered_impact_share']:.1%}, "
            f"affected grids {int(row['total_affected_grids'])}, reports {row['total_reports']:.0f}."
        )
    lines.append("")
    lines.append("## Weakest events")
    for _, row in weak_events.iterrows():
        lines.append(
            f"- {row['city_clean']} Event {int(row['Event_ID'])}: "
            f"uncovered impact {row['uncovered_impact_share']:.1%}, "
            f"affected grids {int(row['total_affected_grids'])}, reports {row['total_reports']:.0f}."
        )
    lines.append("")
    lines.append("## Limitations and next checks")
    lines.append(
        "- The footprint is a reported/socially sensed flood-impact footprint, not an observed inundation polygon."
    )
    lines.append(
        "- The current reproducible run uses leave-one-event-out historical inventories, not a strictly pre-event chronological inventory. "
        "A stricter temporal inventory would require reliable event dates for every event."
    )
    lines.append(
        "- The table is natively 1 km. A 2 km aggregation is included as a diagnostic if requested by the command line; "
        "500 m cannot be recovered from this table without a finer source grid."
    )
    lines.append(
        "- Getis-Ord Gi* validation is not included in this fast national run. It is feasible using the centroid coordinates, "
        "but should be run with explicit spatial weights per city and preferably sparse-neighbor computation/FDR correction."
    )
    lines.append(
        "- Reporting activity bias remains a plausible contributor: POI/population/road-dense places may be more likely to generate reports. "
        "Interpret the gap as a coverage gap in historical public-reporting hotspot inventory."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8-sig")
    progress(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-table", type=Path, default=DEFAULT_FULL_TABLE)
    parser.add_argument("--road-base", type=Path, default=DEFAULT_ROAD_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--include-2km",
        action="store_true",
        help="Also aggregate native grids to 2 km for diagnostic scale sensitivity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full = load_full_table(args.full_table)
    road_lookup, road_meta = load_road_lookup(args.road_base)
    full, merge_meta = attach_road(full, road_lookup)
    audit = build_audit(full, road_meta, merge_meta)

    scales = [PRIMARY_SCALE_M]
    rounded_cell_sizes = set(pd.to_numeric(full["cell_size_m"], errors="coerce").round().dropna().astype(int).unique())
    if args.include_2km:
        scales.append(2000)
    if 500 in rounded_cell_sizes and 500 not in scales:
        scales.append(500)

    all_scale_events = []
    map_grid = None
    for scale_m in scales:
        events, maybe_map = run_scale_analysis(full, scale_m, keep_grid_for_map=(scale_m == PRIMARY_SCALE_M))
        all_scale_events.append(events)
        if maybe_map is not None:
            map_grid = maybe_map

    all_events = pd.concat(all_scale_events, ignore_index=True)
    primary = make_primary_event_table(all_events)
    city = make_city_table(primary)
    sensitivity = make_sensitivity_table(all_events)

    save_csv(primary, args.out_dir / "event_level_coverage_gap.csv")
    save_csv(city, args.out_dir / "city_level_coverage_gap.csv")
    save_csv(sensitivity, args.out_dir / "sensitivity_coverage_gap.csv")

    plot_event_distribution(primary, args.out_dir / "Fig_check_1_event_distribution.png")
    plot_coverage_vs_rainfall(primary, args.out_dir / "Fig_check_2_coverage_vs_rainfall.png")
    plot_exposure_coupling(primary, args.out_dir / "Fig_check_3_exposure_coupling.png")
    if map_grid is not None:
        plot_representative_maps(map_grid, primary, args.out_dir / "Fig_check_4_representative_city_map.png")

    write_technical_note(args.out_dir / "technical_note_coverage_gap.md", primary, city, sensitivity, audit)
    progress("done")


if __name__ == "__main__":
    main()
