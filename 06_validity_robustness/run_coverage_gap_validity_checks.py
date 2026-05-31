#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defensive validity checks for the coverage-gap storyline.

The checks keep the same interpretation boundary as the main analysis:
event-time footprints are reported / socially sensed flood-impact footprints,
not hydrodynamic inundation boundaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

import run_coverage_gap_analysis as cg


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "validity_checks"
PRIMARY_TOP_SHARE = 0.20
PRIMARY_SCALE_M = 1000
REPRESENTATIVE_EVENTS = [
    ("Zhengzhou", "\u90d1\u5dde", 5551),
    ("Guangzhou", "\u5e7f\u5dde", 4520),
    ("Beijing", "\u5317\u4eac", 7098),
    ("Wuhan", "\u6b66\u6c49", 3856),
    ("Chengdu", "\u6210\u90fd", 3215),
]


def progress(msg: str) -> None:
    print(f"[coverage-gap-validity] {msg}", flush=True)


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


def iqr(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.quantile(0.75) - vals.quantile(0.25))


def safe_ratio(numer: float, denom: float) -> float:
    if denom is None or denom <= 0 or pd.isna(denom):
        return np.nan
    return float(numer) / float(denom)


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    if union == 0:
        return np.nan
    return len(a & b) / union


def overlap_coefficient(a: set, b: set) -> float:
    denom = min(len(a), len(b))
    if denom == 0:
        return np.nan
    return len(a & b) / denom


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    valid_idx = np.flatnonzero(valid)
    out[valid_idx[order]] = adjusted
    return out


def load_analysis_table(full_table: Path, road_base: Path) -> pd.DataFrame:
    full = cg.load_full_table(full_table)
    road_lookup, _ = cg.load_road_lookup(road_base)
    full, _ = cg.attach_road(full, road_lookup)
    full["analysis_grid_id"] = full["grid_id"]
    full["analysis_scale_m"] = PRIMARY_SCALE_M
    progress("computing native leave-one-event-out historical reports")
    grid_total = (
        full.groupby(["city_std", "analysis_grid_id"], observed=True)["flood_count"]
        .sum()
        .rename("city_grid_total_reports")
        .reset_index()
    )
    full = full.merge(grid_total, on=["city_std", "analysis_grid_id"], how="left")
    full["hist_reports_loo"] = (
        pd.to_numeric(full["city_grid_total_reports"], errors="coerce").fillna(0)
        - pd.to_numeric(full["flood_count"], errors="coerce").fillna(0)
    ).clip(lower=0)
    return full


def add_primary_inventory_and_summary(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    progress("assigning top20 leave-one-event-out historical inventory for all events")
    hist_flag = cg.assign_historical_inventory(full, PRIMARY_TOP_SHARE)
    full = full.copy()
    full["historical_hotspot_inventory_top20"] = hist_flag.to_numpy(dtype=bool)
    affected = pd.to_numeric(full["flood_count"], errors="coerce").fillna(0) > 0
    progress("summarizing event-level coverage gap for all events")
    summary = cg.summarize_variant(
        full,
        full["historical_hotspot_inventory_top20"],
        affected,
        PRIMARY_TOP_SHARE,
        "gt0",
        PRIMARY_SCALE_M,
    )
    return full, summary


def random_inventory_baseline(
    event_summary: pd.DataFrame,
    full: pd.DataFrame,
    random_reps: int,
    seed: int,
) -> pd.DataFrame:
    progress(f"running random inventory baseline ({random_reps} repetitions)")
    rng = np.random.default_rng(seed)
    extreme = event_summary[event_summary["is_extreme"] == 1].copy()
    city_grid_counts = full.groupby("city_std")["analysis_grid_id"].nunique().to_dict()

    rows = []
    for row in extreme.itertuples(index=False):
        city_grid_n = int(city_grid_counts.get(row.city_std, 0))
        affected_n = int(row.total_affected_grids)
        inventory_n = int(row.historical_hotspot_grids)
        covered_n = int(row.covered_affected_grids)
        sample_n = min(inventory_n, city_grid_n)
        if city_grid_n <= 0 or affected_n <= 0 or sample_n <= 0:
            random_draws = np.zeros(random_reps, dtype=float)
        else:
            random_draws = rng.hypergeometric(
                ngood=affected_n,
                nbad=max(city_grid_n - affected_n, 0),
                nsample=sample_n,
                size=random_reps,
            )
        random_cov = random_draws / affected_n if affected_n > 0 else np.full(random_reps, np.nan)
        random_mean = float(np.nanmean(random_cov)) if np.isfinite(random_cov).any() else np.nan
        observed = float(row.impact_coverage_rate)
        rows.append(
            {
                "city_clean": row.city_clean,
                "city_std": row.city_std,
                "Event_ID": int(row.Event_ID),
                "city_grid_count": city_grid_n,
                "historical_inventory_grids": inventory_n,
                "affected_grids": affected_n,
                "observed_covered_affected_grids": covered_n,
                "observed_impact_coverage_rate": observed,
                "observed_uncovered_impact_share": float(row.uncovered_impact_share),
                "random_mean_impact_coverage_rate": random_mean,
                "random_median_impact_coverage_rate": float(np.nanmedian(random_cov)),
                "random_q05_impact_coverage_rate": float(np.nanquantile(random_cov, 0.05)),
                "random_q95_impact_coverage_rate": float(np.nanquantile(random_cov, 0.95)),
                "coverage_lift": safe_ratio(observed, random_mean),
                "coverage_difference": observed - random_mean,
                "empirical_p_value": float(np.nanmean(random_cov >= observed)),
                "random_reps": random_reps,
            }
        )
    return pd.DataFrame(rows)


def low_sample_robustness(primary_extreme: pd.DataFrame) -> pd.DataFrame:
    progress("computing low-sample robustness filters")
    filters = [
        ("all_events", pd.Series(True, index=primary_extreme.index)),
        ("total_reports_ge_5", primary_extreme["total_reports"] >= 5),
        ("total_reports_ge_10", primary_extreme["total_reports"] >= 10),
        ("total_reports_ge_20", primary_extreme["total_reports"] >= 20),
        ("affected_grids_ge_3", primary_extreme["total_affected_grids"] >= 3),
        ("affected_grids_ge_5", primary_extreme["total_affected_grids"] >= 5),
        ("affected_grids_ge_10", primary_extreme["total_affected_grids"] >= 10),
    ]
    rows = []
    for label, mask in filters:
        sub = primary_extreme[mask].copy()
        gap = sub["uncovered_impact_share"]
        func = sub["function_exposure_outside_historical_hotspots"]
        rows.append(
            {
                "filter": label,
                "n_events": int(len(sub)),
                "median_uncovered_impact_share": float(gap.median()) if len(sub) else np.nan,
                "mean_uncovered_impact_share": float(gap.mean()) if len(sub) else np.nan,
                "median_population_exposure_outside": float(
                    sub["population_exposure_outside_historical_hotspots"].median()
                )
                if len(sub)
                else np.nan,
                "median_road_exposure_outside": float(
                    sub["road_exposure_outside_historical_hotspots"].median()
                )
                if len(sub)
                else np.nan,
                "median_function_exposure_outside": float(func.median()) if len(sub) else np.nan,
                "pearson_gap_function_outside": float(gap.corr(func, method="pearson")) if len(sub) >= 3 else np.nan,
                "spearman_gap_function_outside": float(gap.corr(func, method="spearman")) if len(sub) >= 3 else np.nan,
                "proportion_events_uncovered_share_gt_25": float((gap > 0.25).mean()) if len(sub) else np.nan,
                "proportion_events_uncovered_share_gt_50": float((gap > 0.50).mean()) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def low_history_exclusion_robustness(primary_extreme: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    progress("computing low-history city exclusion robustness")
    city_flags = stability[["city_std", "low_history_city"]].drop_duplicates("city_std")
    work = primary_extreme.merge(city_flags, on="city_std", how="left")
    work["low_history_city"] = work["low_history_city"].fillna(True).astype(bool)
    groups = [
        ("all_extreme_events", work),
        ("non_low_history_cities_only", work[~work["low_history_city"]]),
        ("low_history_cities_only", work[work["low_history_city"]]),
    ]
    rows = []
    for label, sub in groups:
        gap = sub["uncovered_impact_share"]
        func = sub["function_exposure_outside_historical_hotspots"]
        rows.append(
            {
                "sample": label,
                "n_events": int(len(sub)),
                "n_cities": int(sub["city_std"].nunique()),
                "median_uncovered_impact_share": float(gap.median()) if len(sub) else np.nan,
                "mean_uncovered_impact_share": float(gap.mean()) if len(sub) else np.nan,
                "median_population_exposure_outside": float(
                    sub["population_exposure_outside_historical_hotspots"].median()
                )
                if len(sub)
                else np.nan,
                "median_road_exposure_outside": float(
                    sub["road_exposure_outside_historical_hotspots"].median()
                )
                if len(sub)
                else np.nan,
                "median_function_exposure_outside": float(func.median()) if len(sub) else np.nan,
                "pearson_gap_function_outside": float(gap.corr(func, method="pearson")) if len(sub) >= 3 else np.nan,
                "spearman_gap_function_outside": float(gap.corr(func, method="spearman")) if len(sub) >= 3 else np.nan,
                "proportion_events_uncovered_share_gt_25": float((gap > 0.25).mean()) if len(sub) else np.nan,
                "proportion_events_uncovered_share_gt_50": float((gap > 0.50).mean()) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def top_inventory_from_scores(grid_ids: np.ndarray, scores: np.ndarray, top_share: float = PRIMARY_TOP_SHARE) -> set:
    scores = np.asarray(scores, dtype=float)
    grid_ids = np.asarray(grid_ids).astype(str)
    positive = np.flatnonzero(scores > 0)
    if len(positive) == 0:
        return set()
    n_keep = max(1, int(np.ceil(len(positive) * top_share)))
    tmp = pd.DataFrame({"grid_id": grid_ids[positive], "score": scores[positive]})
    tmp = tmp.sort_values(["score", "grid_id"], ascending=[False, True]).head(n_keep)
    return set(tmp["grid_id"].astype(str))


def split_half_stability(full: pd.DataFrame, split_reps: int, seed: int) -> pd.DataFrame:
    progress(f"running split-half historical inventory stability ({split_reps} repetitions)")
    rng = np.random.default_rng(seed)
    pos = full.loc[
        pd.to_numeric(full["flood_count"], errors="coerce").fillna(0) > 0,
        ["city_std", "city_clean", "Event_ID", "analysis_grid_id", "flood_count"],
    ].copy()
    rows = []
    for city_std, city_pos in pos.groupby("city_std", sort=True):
        city_clean = city_pos["city_clean"].dropna().astype(str).iloc[0]
        pivot = city_pos.pivot_table(
            index="analysis_grid_id",
            columns="Event_ID",
            values="flood_count",
            aggfunc="sum",
            fill_value=0,
        )
        events = np.asarray(pivot.columns)
        n_events = len(events)
        low_history = n_events < 4
        j_vals = []
        o_vals = []
        if n_events >= 2:
            arr = pivot.to_numpy(dtype=float)
            grids = pivot.index.astype(str).to_numpy()
            for _ in range(split_reps):
                order = rng.permutation(n_events)
                cut = n_events // 2
                idx_a = order[:cut]
                idx_b = order[cut:]
                if len(idx_a) == 0 or len(idx_b) == 0:
                    continue
                inv_a = top_inventory_from_scores(grids, arr[:, idx_a].sum(axis=1))
                inv_b = top_inventory_from_scores(grids, arr[:, idx_b].sum(axis=1))
                j_vals.append(jaccard(inv_a, inv_b))
                o_vals.append(overlap_coefficient(inv_a, inv_b))
        rows.append(
            {
                "city_clean": city_clean,
                "city_std": city_std,
                "n_historical_events": int(n_events),
                "low_history_city": bool(low_history),
                "mean_jaccard": float(np.nanmean(j_vals)) if len(j_vals) else np.nan,
                "median_jaccard": float(np.nanmedian(j_vals)) if len(j_vals) else np.nan,
                "mean_overlap_coefficient": float(np.nanmean(o_vals)) if len(o_vals) else np.nan,
                "median_overlap_coefficient": float(np.nanmedian(o_vals)) if len(o_vals) else np.nan,
                "split_reps": split_reps,
            }
        )
    all_cities = full[["city_std", "city_clean"]].drop_duplicates("city_std")
    out = all_cities.merge(pd.DataFrame(rows), on=["city_std", "city_clean"], how="left")
    out["n_historical_events"] = out["n_historical_events"].fillna(0).astype(int)
    out["low_history_city"] = out["low_history_city"].fillna(True).astype(bool)
    return out.sort_values(["low_history_city", "median_overlap_coefficient"], ascending=[True, False])


def compute_gistar(city_event_rows: pd.DataFrame) -> pd.DataFrame:
    out = city_event_rows[["analysis_grid_id", "centroid_lon", "centroid_lat", "hist_reports_loo"]].copy()
    out["hist_reports_loo"] = pd.to_numeric(out["hist_reports_loo"], errors="coerce").fillna(0)
    valid = out["centroid_lon"].notna() & out["centroid_lat"].notna()
    out["gistar_z"] = np.nan
    out["gistar_p_value"] = np.nan
    out["gistar_fdr_p_value"] = np.nan
    out["gistar_hotspot"] = False
    if valid.sum() < 5:
        return out

    coords_x, coords_y = cg.lonlat_to_web_mercator(out.loc[valid, "centroid_lon"], out.loc[valid, "centroid_lat"])
    coords = np.column_stack([coords_x, coords_y])
    x = out.loc[valid, "hist_reports_loo"].to_numpy(dtype=float)
    n = len(x)
    xbar = float(x.mean())
    s = float(np.sqrt((x * x).mean() - xbar * xbar))
    if s <= 0 or not np.isfinite(s):
        return out

    tree = cKDTree(coords)
    nn_dists, _ = tree.query(coords, k=min(2, n))
    if nn_dists.ndim == 2:
        median_nn = float(np.nanmedian(nn_dists[:, 1]))
    else:
        median_nn = 1000.0
    radius = max(1500.0, median_nn * 1.5)
    neighbors = tree.query_ball_point(coords, r=radius)

    local_sum = np.array([x[idx].sum() for idx in neighbors], dtype=float)
    sum_w = np.array([len(idx) for idx in neighbors], dtype=float)
    denom = s * np.sqrt((n * sum_w - sum_w * sum_w) / max(n - 1, 1))
    z = np.full(n, np.nan, dtype=float)
    ok = denom > 0
    z[ok] = (local_sum[ok] - xbar * sum_w[ok]) / denom[ok]
    p = norm.sf(z)
    fdr = bh_fdr(p)
    hotspot = (z >= 1.96) & (fdr <= 0.05)

    valid_idx = out.index[valid]
    out.loc[valid_idx, "gistar_z"] = z
    out.loc[valid_idx, "gistar_p_value"] = p
    out.loc[valid_idx, "gistar_fdr_p_value"] = fdr
    out.loc[valid_idx, "gistar_hotspot"] = hotspot
    return out


def compute_gap_with_inventory(event_rows: pd.DataFrame, inventory_set: set, prefix: str) -> dict:
    affected = pd.to_numeric(event_rows["flood_count"], errors="coerce").fillna(0) > 0
    inv = event_rows["analysis_grid_id"].astype(str).isin(inventory_set)
    covered = affected & inv
    uncovered = affected & (~inv)
    reports = pd.to_numeric(event_rows["flood_count"], errors="coerce").fillna(0)
    pop = pd.to_numeric(event_rows["worldpop_2020"], errors="coerce").fillna(0)
    road = pd.to_numeric(event_rows["secondary_road"], errors="coerce").fillna(0)
    func = pd.to_numeric(event_rows["poi_count_2018"], errors="coerce").fillna(0)

    total_affected = int(affected.sum())
    total_reports = float(reports[affected].sum())
    out = {
        f"n_{prefix}_hotspot_grids": int(len(inventory_set)),
        f"covered_affected_grids_using_{prefix}": int(covered.sum()),
        f"uncovered_affected_grids_using_{prefix}": int(uncovered.sum()),
        f"impact_coverage_rate_using_{prefix}": safe_ratio(covered.sum(), total_affected),
        f"uncovered_impact_share_using_{prefix}": safe_ratio(uncovered.sum(), total_affected),
        f"report_weighted_uncovered_share_using_{prefix}": safe_ratio(reports[uncovered].sum(), total_reports),
        f"population_exposure_outside_using_{prefix}": safe_ratio(pop[uncovered].sum(), pop[affected].sum()),
        f"road_exposure_outside_using_{prefix}": safe_ratio(road[uncovered].sum(), road[affected].sum()),
        f"function_exposure_outside_using_{prefix}": safe_ratio(func[uncovered].sum(), func[affected].sum()),
    }
    return out


def gistar_representative_validation(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    progress("running Gi* validation for representative cities")
    rows = []
    map_rows = []
    for label, city_std, event_id in REPRESENTATIVE_EVENTS:
        event_rows = full[(full["city_std"] == city_std) & (full["Event_ID"] == event_id)].copy()
        if event_rows.empty:
            city_events = (
                full[(full["city_std"] == city_std) & (full["is_extreme"] == 1)]
                .groupby("Event_ID")["flood_count"]
                .sum()
                .sort_values(ascending=False)
            )
            if city_events.empty:
                continue
            event_id = int(city_events.index[0])
            event_rows = full[(full["city_std"] == city_std) & (full["Event_ID"] == event_id)].copy()
        top20_set = set(
            event_rows.loc[event_rows["historical_hotspot_inventory_top20"], "analysis_grid_id"].astype(str)
        )
        gistar = compute_gistar(event_rows)
        gistar_set = set(gistar.loc[gistar["gistar_hotspot"], "analysis_grid_id"].astype(str))

        gistar_gap = compute_gap_with_inventory(event_rows, gistar_set, "gistar")
        top20_gap = compute_gap_with_inventory(event_rows, top20_set, "top20")
        row = {
            "panel_city": label,
            "city_clean": event_rows["city_clean"].iloc[0],
            "city_std": city_std,
            "Event_ID": int(event_id),
            "n_total_grids": int(len(event_rows)),
            "n_positive_history_grids": int((event_rows["hist_reports_loo"] > 0).sum()),
            "n_gistar_hotspot_grids": int(len(gistar_set)),
            "n_top20_hotspot_grids": int(len(top20_set)),
            "overlap_with_top20_jaccard": jaccard(gistar_set, top20_set),
            "overlap_with_top20_coefficient": overlap_coefficient(gistar_set, top20_set),
            "max_gistar_z": float(pd.to_numeric(gistar["gistar_z"], errors="coerce").max()),
            "median_gistar_z": float(pd.to_numeric(gistar["gistar_z"], errors="coerce").median()),
        }
        row.update(gistar_gap)
        row.update(top20_gap)
        rows.append(row)

        map_df = event_rows.merge(
            gistar[["analysis_grid_id", "gistar_hotspot", "gistar_z", "gistar_fdr_p_value"]],
            on="analysis_grid_id",
            how="left",
        )
        map_df["panel_city"] = label
        map_df["gistar_hotspot"] = map_df["gistar_hotspot"].fillna(False).astype(bool)
        map_df["top20_hotspot"] = map_df["historical_hotspot_inventory_top20"].astype(bool)
        map_df["event_time_reported_impact"] = pd.to_numeric(map_df["flood_count"], errors="coerce").fillna(0) > 0
        map_df["covered_by_gistar"] = map_df["event_time_reported_impact"] & map_df["gistar_hotspot"]
        map_df["uncovered_by_gistar"] = map_df["event_time_reported_impact"] & (~map_df["gistar_hotspot"])
        map_rows.append(map_df)

    return pd.DataFrame(rows), pd.concat(map_rows, ignore_index=True) if map_rows else pd.DataFrame()


def extreme_vs_nonextreme(event_summary: pd.DataFrame) -> pd.DataFrame:
    progress("computing extreme vs non-extreme comparison")
    rows = []
    for is_extreme, sub in event_summary.groupby("is_extreme"):
        label = "extreme" if int(is_extreme) == 1 else "non_extreme"
        rows.append(
            {
                "event_type": label,
                "is_extreme": int(is_extreme),
                "n_events": int(len(sub)),
                "median_uncovered_impact_share": float(sub["uncovered_impact_share"].median()),
                "iqr_uncovered_impact_share": iqr(sub["uncovered_impact_share"]),
                "median_function_exposure_outside": float(
                    sub["function_exposure_outside_historical_hotspots"].median()
                ),
                "iqr_function_exposure_outside": iqr(sub["function_exposure_outside_historical_hotspots"]),
                "median_report_weighted_uncovered_share": float(
                    sub["report_weighted_uncovered_share"].median()
                ),
                "iqr_report_weighted_uncovered_share": iqr(sub["report_weighted_uncovered_share"]),
                "median_population_exposure_outside": float(
                    sub["population_exposure_outside_historical_hotspots"].median()
                ),
                "median_road_exposure_outside": float(
                    sub["road_exposure_outside_historical_hotspots"].median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("is_extreme")


def plot_random_baseline(random_df: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    vals = [
        random_df["observed_impact_coverage_rate"].dropna().to_numpy() * 100,
        random_df["random_mean_impact_coverage_rate"].dropna().to_numpy() * 100,
    ]
    axes[0].boxplot(vals, patch_artist=True, showfliers=False)
    axes[0].set_xticks([1, 2], ["Historical\ninventory", "Random\ninventory"])
    axes[0].set_ylabel("Impact coverage rate (%)")
    axes[0].set_ylim(-2, 102)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.30)
    lift = random_df["coverage_lift"].replace([np.inf, -np.inf], np.nan).dropna()
    axes[1].hist(lift.clip(upper=10), bins=30, color="#6FA8DC", edgecolor="white")
    axes[1].axvline(1, color="#222222", linewidth=1)
    axes[1].axvline(lift.median(), color="#B23A48", linewidth=1.5)
    axes[1].set_xlabel("Coverage lift vs. random (clipped at 10)")
    axes[1].set_ylabel("Events")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("Historical inventory vs. random inventory baseline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    progress(f"wrote {out_path}")


def plot_random_observed_vs_random(random_df: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    x = random_df["random_mean_impact_coverage_rate"] * 100
    y = random_df["observed_impact_coverage_rate"] * 100
    sig = random_df["empirical_p_value"] < 0.05
    axes[0].scatter(
        x[~sig],
        y[~sig],
        s=18,
        color="#BDBDBD",
        edgecolors="white",
        linewidths=0.25,
        alpha=0.70,
        label="not p<0.05",
    )
    axes[0].scatter(
        x[sig],
        y[sig],
        s=20,
        color="#2C7BB6",
        edgecolors="white",
        linewidths=0.25,
        alpha=0.75,
        label="empirical p<0.05",
    )
    lim = max(100, float(np.nanmax([x.max(), y.max()])))
    axes[0].plot([0, lim], [0, lim], color="#222222", linewidth=1, linestyle="--")
    axes[0].set_xlim(-1, min(lim, 105))
    axes[0].set_ylim(-1, min(lim, 105))
    axes[0].set_xlabel("Random inventory mean coverage (%)")
    axes[0].set_ylabel("Historical inventory coverage (%)")
    axes[0].grid(True, linestyle="--", alpha=0.28)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].text(
        0.03,
        0.97,
        f"median observed={y.median():.1f}%\nmedian random={x.median():.1f}%\np<0.05 share={sig.mean():.1%}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="#BBBBBB", boxstyle="square,pad=0.25", alpha=0.9),
    )

    lift = random_df["coverage_lift"].replace([np.inf, -np.inf], np.nan).dropna()
    lift_pos = lift[lift > 0]
    lift_sorted = np.sort(lift_pos.to_numpy(dtype=float))
    y_ecdf = np.arange(1, len(lift_sorted) + 1) / len(lift_sorted)
    axes[1].plot(lift_sorted, y_ecdf, color="#B23A48", linewidth=1.8)
    axes[1].axvline(1, color="#222222", linewidth=1, linestyle="--")
    axes[1].axvline(lift.median(), color="#B23A48", linewidth=1, alpha=0.75)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Coverage lift vs. random (log scale)")
    axes[1].set_ylabel("Cumulative share of events")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.25)
    axes[1].text(
        0.04,
        0.96,
        f"median lift={lift.median():.2f}x",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="#BBBBBB", boxstyle="square,pad=0.25", alpha=0.9),
    )
    fig.suptitle("Observed historical coverage vs. random baseline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    progress(f"wrote {out_path}")


def plot_low_sample(robustness: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(robustness))
    ax.plot(x, robustness["median_uncovered_impact_share"] * 100, marker="o", label="Uncovered impact")
    ax.plot(x, robustness["median_function_exposure_outside"] * 100, marker="s", label="Function outside")
    ax.plot(x, robustness["median_population_exposure_outside"] * 100, marker="^", label="Population outside")
    ax.axhline(50, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, robustness["filter"], rotation=30, ha="right")
    ax.set_ylabel("Median share (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", alpha=0.30)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.set_title("Low-sample robustness")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    progress(f"wrote {out_path}")


def plot_low_history_exclusion(robustness: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["All", "Non-low\nhistory", "Low\nhistory"]
    x = np.arange(len(robustness))
    ax.bar(
        x - 0.22,
        robustness["median_uncovered_impact_share"] * 100,
        width=0.22,
        color="#2C7BB6",
        label="Uncovered impact",
    )
    ax.bar(
        x,
        robustness["median_function_exposure_outside"] * 100,
        width=0.22,
        color="#F28E2B",
        label="Function outside",
    )
    ax.bar(
        x + 0.22,
        robustness["median_population_exposure_outside"] * 100,
        width=0.22,
        color="#59A14F",
        label="Population outside",
    )
    for i, row in enumerate(robustness.itertuples(index=False)):
        ax.text(i, 3, f"n={row.n_events}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax.axhline(50, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median share (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", alpha=0.30)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.set_title("Coverage gap after excluding low-history cities")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    progress(f"wrote {out_path}")


def plot_split_half(stability: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    main = stability[~stability["low_history_city"]].copy()
    low = stability[stability["low_history_city"]].copy()
    axes[0].boxplot(
        [
            main["median_jaccard"].dropna().to_numpy(),
            low["median_jaccard"].dropna().to_numpy(),
        ],
        patch_artist=True,
        showfliers=False,
    )
    axes[0].set_xticks([1, 2], ["Enough\nhistory", "Low\nhistory"])
    axes[0].set_ylabel("Median Jaccard")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.30)
    axes[1].boxplot(
        [
            main["median_overlap_coefficient"].dropna().to_numpy(),
            low["median_overlap_coefficient"].dropna().to_numpy(),
        ],
        patch_artist=True,
        showfliers=False,
    )
    axes[1].set_xticks([1, 2], ["Enough\nhistory", "Low\nhistory"])
    axes[1].set_ylabel("Median overlap coefficient")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.30)
    fig.suptitle("Split-half stability of historical inventories")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    progress(f"wrote {out_path}")


def plot_gistar_maps(map_df: pd.DataFrame, validation: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    if map_df.empty:
        return
    panels = validation["panel_city"].tolist()
    fig, axes = plt.subplots(1, len(panels), figsize=(14.5, 3.8))
    if len(panels) == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        sub = map_df[map_df["panel_city"] == panel].copy()
        row = validation[validation["panel_city"] == panel].iloc[0]
        show = sub[sub["top20_hotspot"] | sub["gistar_hotspot"] | sub["event_time_reported_impact"]].copy()
        lon = pd.to_numeric(show["centroid_lon"], errors="coerce")
        lat = pd.to_numeric(show["centroid_lat"], errors="coerce")
        lon_span = max(float(lon.max() - lon.min()), 0.02)
        lat_span = max(float(lat.max() - lat.min()), 0.02)
        ax.set_xlim(float(lon.min() - lon_span * 0.08), float(lon.max() + lon_span * 0.08))
        ax.set_ylim(float(lat.min() - lat_span * 0.08), float(lat.max() + lat_span * 0.08))

        top20 = show[show["top20_hotspot"]]
        gistar = show[show["gistar_hotspot"]]
        covered = show[show["covered_by_gistar"]]
        uncovered = show[show["uncovered_by_gistar"]]
        reports = show[show["event_time_reported_impact"]]
        ax.scatter(top20["centroid_lon"], top20["centroid_lat"], marker="s", s=18, color="#D9D9D9", edgecolors="#999999", linewidths=0.2, alpha=0.7, label="Top20 inventory")
        ax.scatter(gistar["centroid_lon"], gistar["centroid_lat"], marker="s", s=30, color="#7B3294", edgecolors="white", linewidths=0.25, alpha=0.85, label="Gi* inventory")
        ax.scatter(covered["centroid_lon"], covered["centroid_lat"], marker="o", s=24, color="#2C7BB6", edgecolors="white", linewidths=0.2, alpha=0.85, label="Covered impact")
        ax.scatter(uncovered["centroid_lon"], uncovered["centroid_lat"], marker="o", s=24, color="#D7191C", edgecolors="white", linewidths=0.2, alpha=0.75, label="Uncovered impact")
        ax.scatter(reports["centroid_lon"], reports["centroid_lat"], s=8, color="#111111", alpha=0.35, linewidths=0, label="Flood reports")
        ax.set_title(
            f"{panel}\nGi* gap {row['uncovered_impact_share_using_gistar']:.0%}; top20 {row['uncovered_impact_share_using_top20']:.0%}",
            fontsize=8.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.45)
            spine.set_color("#BBBBBB")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Representative Gi* validation maps", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    progress(f"wrote {out_path}")


def plot_extreme_vs_nonextreme(event_summary: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    data = event_summary.copy()
    data["event_type"] = np.where(data["is_extreme"] == 1, "Extreme", "Non-extreme")
    metrics = [
        ("uncovered_impact_share", "Uncovered impact"),
        ("function_exposure_outside_historical_hotspots", "Function outside"),
        ("report_weighted_uncovered_share", "Report-weighted outside"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        vals = [
            data.loc[data["event_type"] == "Non-extreme", col].dropna().to_numpy() * 100,
            data.loc[data["event_type"] == "Extreme", col].dropna().to_numpy() * 100,
        ]
        ax.boxplot(vals, patch_artist=True, showfliers=False)
        ax.set_xticks([1, 2], ["Non-\nextreme", "Extreme"])
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-2, 102)
        ax.grid(True, axis="y", linestyle="--", alpha=0.30)
    axes[0].set_ylabel("Share (%)")
    fig.suptitle("Coverage gap in extreme vs. non-extreme events")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    progress(f"wrote {out_path}")


def write_technical_note(
    out_path: Path,
    random_df: pd.DataFrame,
    low_sample: pd.DataFrame,
    low_history_exclusion: pd.DataFrame,
    stability: pd.DataFrame,
    gistar: pd.DataFrame,
    extreme_cmp: pd.DataFrame,
) -> None:
    rand_obs = random_df["observed_impact_coverage_rate"].median()
    rand_mean = random_df["random_mean_impact_coverage_rate"].median()
    lift = random_df["coverage_lift"].replace([np.inf, -np.inf], np.nan).median()
    sig_share = (random_df["empirical_p_value"] < 0.05).mean()
    low_min_gap = low_sample["median_uncovered_impact_share"].min()
    low_min_func = low_sample["median_function_exposure_outside"].min()
    non_low = low_history_exclusion[low_history_exclusion["sample"] == "non_low_history_cities_only"].iloc[0]
    low_only = low_history_exclusion[low_history_exclusion["sample"] == "low_history_cities_only"].iloc[0]
    main_stable = stability[~stability["low_history_city"]].copy()
    med_j = main_stable["median_jaccard"].median()
    med_o = main_stable["median_overlap_coefficient"].median()
    low_hist_share = stability["low_history_city"].mean()
    gistar_gap = gistar["uncovered_impact_share_using_gistar"].median() if not gistar.empty else np.nan
    top20_gap_rep = gistar["uncovered_impact_share_using_top20"].median() if not gistar.empty else np.nan
    gistar_overlap = gistar["overlap_with_top20_coefficient"].median() if not gistar.empty else np.nan

    ext = extreme_cmp[extreme_cmp["event_type"] == "extreme"].iloc[0]
    non = extreme_cmp[extreme_cmp["event_type"] == "non_extreme"].iloc[0]
    ext_gap = float(ext["median_uncovered_impact_share"])
    non_gap = float(non["median_uncovered_impact_share"])
    ext_func = float(ext["median_function_exposure_outside"])
    non_func = float(non["median_function_exposure_outside"])

    criteria = [
        rand_obs > rand_mean and sig_share > 0.50,
        1 - rand_obs > 0.50,
        low_min_gap > 0.50,
        med_o > 0.30,
        gistar_gap > 0.50,
        ext_gap >= non_gap or ext_func >= non_func,
        low_min_func > 0.50,
    ]
    passed = sum(bool(x) for x in criteria)
    verdict = "strong support" if passed >= 5 else "moderate support" if passed >= 3 else "weak support"

    lines = []
    lines.append("# Technical note: coverage-gap validity checks")
    lines.append("")
    lines.append("## 1. Random inventory baseline")
    lines.append(
        f"- Median observed impact coverage by historical inventory is {rand_obs:.1%}, compared with random inventory {rand_mean:.1%}."
    )
    lines.append(
        f"- Median coverage lift is {lift:.2f}x; {sig_share:.1%} of events have empirical p < 0.05 for historical coverage exceeding random coverage."
    )
    lines.append(
        "- Interpretation: historical inventories are informative relative to random inventories, but their absolute coverage remains low."
    )
    lines.append("")
    lines.append("## 2. Low-sample event robustness")
    lines.append(
        f"- Across all low-sample filters, the minimum median uncovered impact share is {low_min_gap:.1%}; "
        f"the minimum median function exposure outside is {low_min_func:.1%}."
    )
    for row in low_sample.itertuples(index=False):
        lines.append(
            f"- {row.filter}: n={row.n_events}, median gap={row.median_uncovered_impact_share:.1%}, "
            f"median function outside={row.median_function_exposure_outside:.1%}, "
            f"Pearson/Spearman={row.pearson_gap_function_outside:.2f}/{row.spearman_gap_function_outside:.2f}."
        )
    lines.append("")
    lines.append("## 3. Split-half historical inventory stability")
    lines.append(
        f"- Among cities with enough history, median split-half Jaccard is {med_j:.2f}; "
        f"median overlap coefficient is {med_o:.2f}."
    )
    lines.append(
        f"- Low-history cities account for {low_hist_share:.1%} of cities and should be flagged or down-weighted in narrative examples."
    )
    lines.append(
        f"- Excluding low-history cities leaves {int(non_low.n_events)} extreme events across {int(non_low.n_cities)} cities; "
        f"median uncovered impact remains {non_low.median_uncovered_impact_share:.1%}, "
        f"with median function exposure outside {non_low.median_function_exposure_outside:.1%}."
    )
    lines.append(
        f"- Low-history cities alone have median uncovered impact {low_only.median_uncovered_impact_share:.1%}; "
        "this means they are not the sole source of the national coverage-gap result."
    )
    lines.append("")
    lines.append("## 4. Gi* representative-city validation")
    if gistar.empty:
        lines.append("- Gi* validation did not produce representative rows.")
    else:
        lines.append(
            f"- Median Gi*-based uncovered impact share is {gistar_gap:.1%}, versus top20 median {top20_gap_rep:.1%} for the same representative events."
        )
        lines.append(
            f"- Median Gi*/top20 overlap coefficient is {gistar_overlap:.2f}."
        )
        for row in gistar.itertuples(index=False):
            lines.append(
                f"- {row.panel_city} Event {row.Event_ID}: Gi* hotspots={row.n_gistar_hotspot_grids}, "
                f"top20 hotspots={row.n_top20_hotspot_grids}, overlap coefficient={row.overlap_with_top20_coefficient:.2f}, "
                f"Gi* gap={row.uncovered_impact_share_using_gistar:.1%}, top20 gap={row.uncovered_impact_share_using_top20:.1%}."
            )
    lines.append("")
    lines.append("## 5. Extreme vs non-extreme comparison")
    lines.append(
        f"- Extreme events: median uncovered impact {ext_gap:.1%}, median function outside {ext_func:.1%}."
    )
    lines.append(
        f"- Non-extreme events: median uncovered impact {non_gap:.1%}, median function outside {non_func:.1%}."
    )
    if ext_gap > non_gap:
        lines.append("- Extreme events show a higher median spatial coverage gap than non-extreme events.")
    else:
        lines.append(
            "- Extreme events are not higher on median spatial gap than non-extreme events; the safer phrasing is that historical inventories are generally incomplete, with severe implications under extreme rainfall."
        )
    lines.append("")
    lines.append("## Final judgement")
    lines.append(f"- Passed {passed}/7 defensive criteria; final judgement: **{verdict}**.")
    if verdict == "strong support":
        lines.append("- Recommendation: formally shift the paper mainline to historical hotspot inventory coverage gap.")
    elif verdict == "moderate support":
        lines.append("- Recommendation: the coverage-gap mainline is viable, but present it with caveats around weak checks.")
    else:
        lines.append("- Recommendation: keep coverage gap as a supporting analysis rather than the mainline.")
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append(
        "- All footprints are public-reporting / socially sensed flood-impact footprints, not observed inundation polygons."
    )
    lines.append(
        "- Inventories are leave-one-event-out rather than strictly pre-event chronological; a chronology-only check remains a useful next robustness step if reliable event dates are harmonized."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8-sig")
    progress(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-table", type=Path, default=cg.DEFAULT_FULL_TABLE)
    parser.add_argument("--road-base", type=Path, default=cg.DEFAULT_ROAD_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--random-reps", type=int, default=500)
    parser.add_argument("--split-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=416)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full = load_analysis_table(args.full_table, args.road_base)
    full, event_summary = add_primary_inventory_and_summary(full)
    primary_extreme = event_summary[event_summary["is_extreme"] == 1].copy()

    random_df = random_inventory_baseline(primary_extreme, full, args.random_reps, args.seed)
    low_sample = low_sample_robustness(primary_extreme)
    stability = split_half_stability(full, args.split_reps, args.seed + 1)
    low_history_exclusion = low_history_exclusion_robustness(primary_extreme, stability)
    gistar, gistar_map = gistar_representative_validation(full)
    extreme_cmp = extreme_vs_nonextreme(event_summary)

    save_csv(random_df, args.out_dir / "random_inventory_baseline.csv")
    save_csv(low_sample, args.out_dir / "low_sample_robustness.csv")
    save_csv(stability, args.out_dir / "historical_inventory_split_half_stability.csv")
    save_csv(low_history_exclusion, args.out_dir / "low_history_exclusion_robustness.csv")
    save_csv(gistar, args.out_dir / "gistars_representative_city_validation.csv")
    save_csv(extreme_cmp, args.out_dir / "extreme_vs_nonextreme_coverage_gap.csv")

    plot_random_baseline(random_df, args.out_dir / "Fig_validity_1_random_baseline.png")
    plot_random_observed_vs_random(random_df, args.out_dir / "Fig_validity_1b_random_observed_vs_random.png")
    plot_low_sample(low_sample, args.out_dir / "Fig_validity_2_low_sample_filters.png")
    plot_split_half(stability, args.out_dir / "Fig_validity_3_split_half_stability.png")
    plot_low_history_exclusion(low_history_exclusion, args.out_dir / "Fig_validity_3b_low_history_exclusion.png")
    plot_gistar_maps(gistar_map, gistar, args.out_dir / "Fig_validity_4_gistar_validation_maps.png")
    plot_extreme_vs_nonextreme(event_summary, args.out_dir / "Fig_validity_5_extreme_vs_nonextreme.png")

    write_technical_note(
        args.out_dir / "technical_note_validity_checks.md",
        random_df,
        low_sample,
        low_history_exclusion,
        stability,
        gistar,
        extreme_cmp,
    )
    progress("done")


if __name__ == "__main__":
    main()
