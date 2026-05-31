#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


EVENT_ID_CANDIDATES = ["Event_ID", "event_id", "eventid", "eventID"]
CITY_CANDIDATES = ["city_clean", "city", "city_std"]
GRID_CANDIDATES = ["grid_id", "gridid", "grid"]
EXTREME_CANDIDATES = ["is_extreme", "extreme", "isExtreme"]
HOTSPOT_CANDIDATES = ["hotspot_refined", "hotspot", "is_hotspot"]
NEW_CANDIDATES = ["new_hotspot_region", "NewHotspot_ge", "new_hotspot", "newly_opened_hotspot"]
DATE_CANDIDATES = [
    "event_start_date", "Event_Start", "event_start", "start_time", "start_date",
    "event_date", "date", "post_date", "tm", "timestamp",
    "first_flood_time", "last_flood_time",
    "Post_Time_min", "Post_Time_max",
    "Event_Peak_Time",
]
YEAR_CANDIDATES = ["event_year", "Event_Year", "year", "report_year", "start_year"]

COVARIATE_CANDIDATES = {
    "distance_to_centre": [
        "distance_to_centre", "distance_to_center", "dist_to_center", "distcenter",
        "dist_to_center_km", "distance_to_center_km", "dist2center_km",
    ],
    "night_time_lights": [
        "night_time_lights", "ntl", "viirs_avg_rad", "avg_rad",
        "ntl_avg_rad_2021", "ntl_avg_rad", "viirs_avg_rad_2021",
    ],
    "population": ["population", "worldpop", "worldpop_2020", "pop_2020", "pop"],
    "poi_log1p": ["poi_log1p", "poi_count_log1p", "log1p_poi_count", "poi_log1p_2018"],
    "poi_count": ["poi_count", "poi_total", "poi", "poi_2018", "poi_count_2018"],
    "poi_category_richness": ["poi_category_richness", "poi_richness", "poi_cat_richness", "poi_unique_cat_n_2018"],
    "twi": ["twi", "topographic_wetness_index", "wetness_index", "twi_proxy"],
    "elevation": ["elevation", "dem", "elev", "elevation_copdem_m"],
    "slope": ["slope", "slope_copdem_deg"],
    "development_year": ["development_year", "gaia_development_year", "gaia_year", "devyear_from_gaia"],
}


def find_first_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not infer a column from candidates: {list(candidates)}")
    return None


def infer_basic_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {
        "event_id": find_first_column(df, EVENT_ID_CANDIDATES),
        "city": find_first_column(df, CITY_CANDIDATES),
        "grid": find_first_column(df, GRID_CANDIDATES),
        "is_extreme": find_first_column(df, EXTREME_CANDIDATES),
        "hotspot": find_first_column(df, HOTSPOT_CANDIDATES),
        "new_hotspot": find_first_column(df, NEW_CANDIDATES, required=False),
    }


def infer_covariates(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    out = {}
    for key, candidates in COVARIATE_CANDIDATES.items():
        out[key] = find_first_column(df, candidates, required=False)
    return out


def infer_event_time(df: pd.DataFrame) -> pd.DataFrame:
    basic = infer_basic_columns(df)
    event_col = basic["event_id"]
    date_col = find_first_column(df, DATE_CANDIDATES, required=False)
    year_col = find_first_column(df, YEAR_CANDIDATES, required=False)

    meta = df[[event_col]].drop_duplicates().copy()

    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        tmp = pd.DataFrame({event_col: df[event_col], "event_time": dt})
        tmp = tmp.dropna(subset=["event_time"])
        if not tmp.empty:
            meta = meta.merge(tmp.groupby(event_col, as_index=False)["event_time"].min(), on=event_col, how="left")
            meta["event_year"] = meta["event_time"].dt.year
            return meta

    if year_col is not None:
        yr = pd.to_numeric(df[year_col], errors="coerce")
        tmp = pd.DataFrame({event_col: df[event_col], "event_year": yr})
        tmp = tmp.dropna(subset=["event_year"])
        if not tmp.empty:
            meta = meta.merge(tmp.groupby(event_col, as_index=False)["event_year"].median(), on=event_col, how="left")
            meta["event_year"] = meta["event_year"].round().astype("Int64")
            return meta

    # Fallback: try to recover event-year from event-meta tables commonly stored
    # in refined outputs, then map back by Event_ID.
    fallback = infer_event_time_from_event_meta(df[event_col].dropna().unique())
    if fallback is not None and not fallback.empty:
        meta = meta.merge(fallback, on=event_col, how="left")
        if meta["event_year"].notna().any():
            return meta

    raise ValueError(
        "Could not infer event date/year from the input table or event-meta fallback."
    )


def infer_event_time_from_event_meta(event_ids) -> Optional[pd.DataFrame]:
    # Support both run modes:
    # 1) from repository root, 2) from code416 subfolder.
    repo_root = Path(__file__).resolve().parents[1]
    roots = [Path.cwd(), repo_root]
    rel_files = [
        Path("outputs_spatial_burden/refined_outputs/event_meta_with_extreme_flags_builtup.csv"),
        Path("outputs_spatial_burden/refined_outputs/event_meta_with_extreme_flags_region.csv"),
        Path("outputs_spatial_burden/refined_outputs/event_meta_screened_with_peak_builtup.csv"),
        Path("outputs_spatial_burden/refined_outputs/event_meta_all.csv"),
    ]
    candidates = []
    for root in roots:
        for rel in rel_files:
            candidates.append((root / rel).resolve())
    for fp in candidates:
        if not fp.exists():
            continue
        try:
            meta = pd.read_csv(fp, low_memory=False)
        except Exception:
            continue
        meta.columns = [str(c).strip() for c in meta.columns]
        event_col = find_first_column(meta, EVENT_ID_CANDIDATES, required=False)
        if event_col is None:
            continue
        date_col = find_first_column(meta, DATE_CANDIDATES, required=False)
        year_col = find_first_column(meta, YEAR_CANDIDATES, required=False)

        out = pd.DataFrame({event_col: pd.Series(event_ids)})
        if date_col is not None and date_col in meta.columns:
            dt = pd.to_datetime(meta[date_col], errors="coerce")
            tmp = pd.DataFrame({event_col: meta[event_col], "event_time": dt}).dropna(subset=["event_time"])
            if not tmp.empty:
                out = out.merge(tmp.groupby(event_col, as_index=False)["event_time"].min(), on=event_col, how="left")
                out["event_year"] = out["event_time"].dt.year
                return out[[event_col, "event_year"]]

        if year_col is not None and year_col in meta.columns:
            yr = pd.to_numeric(meta[year_col], errors="coerce")
            tmp = pd.DataFrame({event_col: meta[event_col], "event_year": yr}).dropna(subset=["event_year"])
            if not tmp.empty:
                out = out.merge(tmp.groupby(event_col, as_index=False)["event_year"].median(), on=event_col, how="left")
                out["event_year"] = out["event_year"].round().astype("Int64")
                return out[[event_col, "event_year"]]
    return None


def ensure_numeric(df: pd.DataFrame, cols: Iterable[Optional[str]]) -> pd.DataFrame:
    for c in cols:
        if c is not None and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


@dataclass
class ModelResult:
    model: str
    outcome: str
    cluster: str
    n: int
    events: int
    cities: int
    term: str
    coef: float
    se: float


def run_event_fe_lpm(
    df: pd.DataFrame,
    outcome_col: str,
    predictors: list[str],
    event_col: str,
    city_col: str,
    cluster_col: str = "event",
    model_name: str = "M2",
) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except Exception as e:
        raise ImportError("statsmodels is required for the robustness scripts. Please install it first.") from e

    use_cols = [outcome_col, event_col, city_col] + predictors
    work = df[use_cols].copy()
    work = work.dropna(subset=[outcome_col] + predictors)
    if work.empty:
        raise ValueError(f"No observations left after dropping NA for model {model_name}.")

    grouped = work.groupby(event_col)
    y_dm = work[outcome_col] - grouped[outcome_col].transform("mean")
    X_dm = pd.DataFrame(index=work.index)
    for p in predictors:
        X_dm[p] = work[p] - grouped[p].transform("mean")

    keep = X_dm.notna().all(axis=1) & y_dm.notna()
    X_dm = X_dm.loc[keep]
    y_dm = y_dm.loc[keep]
    work = work.loc[keep]

    fit = sm.OLS(y_dm, X_dm).fit()

    if cluster_col == "event":
        groups = work[event_col]
    elif cluster_col == "city":
        groups = work[city_col]
    else:
        groups = work[cluster_col] if cluster_col in work.columns else work[event_col]

    rob = fit.get_robustcov_results(cov_type="cluster", groups=groups)

    rows = []
    for i, term in enumerate(X_dm.columns):
        rows.append(
            ModelResult(
                model=model_name,
                outcome=outcome_col,
                cluster=cluster_col,
                n=int(len(work)),
                events=int(work[event_col].nunique()),
                cities=int(work[city_col].nunique()),
                term=term,
                coef=float(rob.params[i]),
                se=float(rob.bse[i]),
            ).__dict__
        )
    return pd.DataFrame(rows)


def prepare_full_footprint_sample(
    df: pd.DataFrame,
    outcome_col: str,
    cutoff_year: Optional[int] = None,
) -> tuple[pd.DataFrame, Dict[str, str], Dict[str, Optional[str]]]:
    basic = infer_basic_columns(df)
    covs = infer_covariates(df)
    time_meta = infer_event_time(df)

    df = df.copy()
    df = ensure_numeric(
        df,
        [basic["is_extreme"], basic["hotspot"], basic["new_hotspot"]]
        + [v for v in covs.values() if v is not None]
        + [outcome_col],
    )
    df = df.merge(time_meta, on=basic["event_id"], how="left")

    sample = df[df[basic["is_extreme"]] == 1].copy()
    if cutoff_year is not None:
        sample = sample[sample["event_year"].astype("float") >= float(cutoff_year)].copy()

    for key in ["distance_to_centre", "night_time_lights", "population", "twi", "elevation", "slope", "development_year"]:
        col = covs.get(key)
        if col is not None and col in sample.columns:
            sample[f"z_{key}"] = zscore(sample[col])

    if covs.get("poi_log1p") is not None and covs["poi_log1p"] in sample.columns:
        sample["z_poi_log1p"] = zscore(sample[covs["poi_log1p"]])
    elif covs.get("poi_count") is not None and covs["poi_count"] in sample.columns:
        sample["z_poi_log1p"] = zscore(np.log1p(pd.to_numeric(sample[covs["poi_count"]], errors="coerce")))
    else:
        sample["z_poi_log1p"] = np.nan

    if covs.get("poi_category_richness") is not None and covs["poi_category_richness"] in sample.columns:
        sample["z_poi_category_richness"] = zscore(sample[covs["poi_category_richness"]])
    else:
        sample["z_poi_category_richness"] = np.nan

    sample[outcome_col] = pd.to_numeric(sample[outcome_col], errors="coerce")
    return sample, basic, covs


def get_model_specs(sample: pd.DataFrame) -> Dict[str, list[str]]:
    terrain = [c for c in ["z_twi", "z_elevation", "z_slope", "z_development_year"] if c in sample.columns and sample[c].notna().any()]
    base = terrain + [c for c in ["z_distance_to_centre", "z_night_time_lights", "z_population"] if c in sample.columns and sample[c].notna().any()]
    m2 = base
    m3 = base + (["z_poi_log1p"] if "z_poi_log1p" in sample.columns and sample["z_poi_log1p"].notna().any() else [])
    m4 = base + (["z_poi_category_richness"] if "z_poi_category_richness" in sample.columns and sample["z_poi_category_richness"].notna().any() else [])
    return {"M2": m2, "M3": m3, "M4": m4}


def summarize_positive_rate(sample: pd.DataFrame, outcome_col: str, event_col: str, city_col: str) -> dict:
    y = pd.to_numeric(sample[outcome_col], errors="coerce")
    return {
        "n_obs": int(y.notna().sum()),
        "positive_n": int((y == 1).sum()),
        "positive_rate": float((y == 1).mean()),
        "events": int(sample[event_col].nunique()),
        "cities": int(sample[city_col].nunique()),
    }


def build_recent_window_outcome(
    df: pd.DataFrame,
    lookback_years: int,
    output_col: str,
) -> pd.DataFrame:
    basic = infer_basic_columns(df)
    time_meta = infer_event_time(df)

    work = df.copy().merge(time_meta, on=basic["event_id"], how="left")
    work = ensure_numeric(work, [basic["is_extreme"], basic["hotspot"]])

    event_col = basic["event_id"]
    city_col = basic["city"]
    grid_col = basic["grid"]
    extreme_col = basic["is_extreme"]
    hotspot_col = basic["hotspot"]

    work[output_col] = 0

    candidate = work[(work[extreme_col] == 1) & (work[hotspot_col] == 1)].copy()
    history = work[(work[extreme_col] == 0) & (work[hotspot_col] == 1)].copy()

    if candidate.empty:
        return work

    has_datetime = "event_time" in work.columns and work["event_time"].notna().any()

    hist_groups = {}
    if has_datetime:
        history = history.dropna(subset=["event_time"])
        for (city, grid), sub in history.groupby([city_col, grid_col]):
            hist_groups[(city, grid)] = np.sort(sub["event_time"].values.astype("datetime64[ns]"))
        window_days = int(round(365.25 * lookback_years))

        def has_recent_history(row):
            key = (row[city_col], row[grid_col])
            arr = hist_groups.get(key)
            if arr is None or pd.isna(row["event_time"]):
                return False
            t = np.datetime64(row["event_time"])
            low = t - np.timedelta64(window_days, "D")
            return bool(((arr < t) & (arr >= low)).any())
    else:
        history = history.dropna(subset=["event_year"])
        for (city, grid), sub in history.groupby([city_col, grid_col]):
            hist_groups[(city, grid)] = np.sort(sub["event_year"].astype(float).values)

        def has_recent_history(row):
            key = (row[city_col], row[grid_col])
            arr = hist_groups.get(key)
            if arr is None or pd.isna(row["event_year"]):
                return False
            yr = float(row["event_year"])
            low = yr - float(lookback_years)
            return bool(((arr < yr) & (arr >= low)).any())

    recent_hist = candidate.apply(has_recent_history, axis=1)
    candidate[output_col] = (~recent_hist).astype(int)
    work.loc[candidate.index, output_col] = candidate[output_col].values
    return work


def write_text_report(path: str, title: str, summaries: list[dict], note: str = "") -> None:
    lines = [title, "=" * len(title), ""]
    for s in summaries:
        lines.extend([
            f"Sample: {s.get('sample_name', '')}",
            f"  N obs: {s['n_obs']}",
            f"  Positive N: {s['positive_n']}",
            f"  Positive rate: {s['positive_rate']:.6f}",
            f"  Events: {s['events']}",
            f"  Cities: {s['cities']}",
            "",
        ])
    if note:
        lines.append(note)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
