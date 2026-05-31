#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
City heterogeneity checks for the reframed narrative.

What this script does:
1) builds city built-up area quartiles from the static 253-city grid base
2) computes city-average new-hotspot shares across extreme events
3) re-estimates full-sample event-FE models by built-up quartile (devyear removed)
4) adds an event-level interaction model: peak rainfall x city built-up area
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster

DEFAULT_BASE = Path("city_grid_base_centroids_gaia_with_gee_terrain_repaired_253cities.csv")
DEFAULT_FULL = Path("city_event_grid_full_gaia_v4_poi_repaired_253cities.csv")


def resolve_input_path(user_arg: str, default_name: str) -> Path:
    """Resolve an input CSV path robustly across common project layouts.

    Order:
    1) exact user-provided path
    2) relative to current working directory
    3) relative to the script directory
    4) relative to likely project roots near the script / cwd
    """
    p = Path(user_arg)
    if p.exists():
        return p.resolve()

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    candidates = []
    if not p.is_absolute():
        candidates.extend([
            cwd / p,
            script_dir / p,
            cwd.parent / p,
            script_dir.parent / p,
        ])

    project_roots = []
    for root in [cwd, cwd.parent, script_dir, script_dir.parent, script_dir.parent.parent if script_dir.parent != script_dir else script_dir]:
        if root not in project_roots:
            project_roots.append(root)

    common_subdirs = [
        Path('outputs_spatial_burden/refined_outputs/grid_universe/poi2018_outputs'),
        Path('outputs_spatial_burden/refined_outputs/grid_universe'),
        Path('outputs_spatial_burden/refined_outputs'),
        Path('refined_outputs/grid_universe/poi2018_outputs'),
        Path('refined_outputs/grid_universe'),
        Path('refined_outputs'),
        Path('.'),
    ]
    for root in project_roots:
        for sub in common_subdirs:
            candidates.append(root / sub / default_name)
            if not p.is_absolute():
                candidates.append(root / sub / p.name)

    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand.resolve()

    msg = [f"Missing input file: {user_arg}", "Tried common locations including:"]
    for cand in list(dict.fromkeys(str(c) for c in candidates[:12])):
        msg.append(f"  - {cand}")
    RAISEPLACEHOLDER


BASE_VARS = ["twi", "elev", "slope"]
CENTER_ACTIVITY_VARS = ["ntl", "pop", "distcenter"]
POI_LOG_VAR = "poi_log"
POI_CAT_VAR = "poi_cat"


def normalize_city(x):
    if pd.isna(x):
        return ""
    return str(x).strip().replace(" ", "")


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def sign_star(coef, p):
    if pd.isna(coef):
        return ""
    sign = "+" if coef > 0 else ("-" if coef < 0 else "0")
    return sign + stars(p)


def city_builtup_area(base_df: pd.DataFrame) -> pd.DataFrame:
    x = base_df.copy()
    x["city_clean"] = x["city_clean"].map(normalize_city)
    x["cell_size_m"] = pd.to_numeric(x["cell_size_m"], errors="coerce").fillna(1000)
    by_city = (
        x.groupby("city_clean", as_index=False)
         .agg(n_builtup_grids=("grid_id", "nunique"), cell_size_m=("cell_size_m", "median"))
    )
    by_city["builtup_area_km2"] = by_city["n_builtup_grids"] * (by_city["cell_size_m"] ** 2) / 1_000_000.0
    ranks = by_city["builtup_area_km2"].rank(method="average", pct=True)
    by_city["builtup_area_quartile"] = pd.cut(
        ranks,
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"],
        include_lowest=True,
    ).astype(str)
    return by_city


def build_event_city_tables(full_df: pd.DataFrame):
    x = full_df.copy()
    x["city_clean"] = x["city_clean"].map(normalize_city)
    x["is_extreme"] = pd.to_numeric(x["is_extreme"], errors="coerce")
    x["new_hotspot_region"] = pd.to_numeric(x["new_hotspot_region"], errors="coerce").fillna(0)
    x["flood_count"] = pd.to_numeric(x["flood_count"], errors="coerce").fillna(0)
    x["Event_Peak_Rain"] = pd.to_numeric(x["Event_Peak_Rain"], errors="coerce")
    x = x[x["is_extreme"] == 1].copy()

    event = (
        x.groupby(["city_clean", "Event_ID"], as_index=False)
         .agg(
             event_new_hotspot_share=("new_hotspot_region", "mean"),
             new_hotspot_grids=("new_hotspot_region", "sum"),
             n_event_grids=("grid_id", "nunique"),
             event_burden=("event_burden", "max"),
             peak_rain=("Event_Peak_Rain", "max"),
         )
    )
    city_avg = (
        event.groupby("city_clean", as_index=False)
             .agg(
                 city_avg_new_hotspot_share=("event_new_hotspot_share", "mean"),
                 city_median_new_hotspot_share=("event_new_hotspot_share", "median"),
                 n_extreme_events=("Event_ID", "nunique"),
             )
    )
    return event, city_avg


def make_boxplot(df: pd.DataFrame, out_png: Path):
    order = ["Q1 smallest", "Q2", "Q3", "Q4 largest"]
    grouped = []
    labels = []
    for g in order:
        vals = df.loc[df["builtup_area_quartile"].astype(str) == g, "city_avg_new_hotspot_share"].dropna().values
        if len(vals) > 0:
            grouped.append(vals)
            labels.append(g)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.boxplot(grouped, labels=labels)
    ax.set_title("City-average new-hotspot share by built-up area quartile", fontsize=12)
    ax.set_ylabel("City-average new-hotspot share")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def fit_event_fe_lpm_citycluster(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    use = df[[y_col, "Event_ID", "city_clean"] + x_cols].dropna().copy()
    if use.empty or use[y_col].nunique() < 2:
        return None, None, None, {"status": "fail"}

    use[y_col + "_dm"] = use[y_col] - use.groupby("Event_ID")[y_col].transform("mean")
    x_dm_cols = []
    for x in x_cols:
        xdm = x + "_dm"
        use[xdm] = use[x] - use.groupby("Event_ID")[x].transform("mean")
        x_dm_cols.append(xdm)
    keep = [y_col + "_dm"] + x_dm_cols + ["Event_ID", "city_clean"]
    use = use[keep].dropna().copy()

    row_nonzero = np.zeros(len(use), dtype=bool)
    for c in [y_col + "_dm"] + x_dm_cols:
        row_nonzero |= (np.abs(use[c].to_numpy()) > 0)
    use = use.loc[row_nonzero].copy()
    if use.empty:
        return None, None, None, {"status": "fail"}

    X = use[x_dm_cols].astype(float)
    y = use[y_col + "_dm"].astype(float)
    model = sm.OLS(y, X).fit()
    city_codes = pd.factorize(use["city_clean"], sort=False)[0].astype(np.int64)
    cov = cov_cluster(model, city_codes)
    se = pd.Series(np.sqrt(np.diag(cov)), index=model.params.index)
    z = pd.Series(model.params / se, index=model.params.index)
    p = pd.Series(2 * (1 - stats.norm.cdf(np.abs(z))), index=model.params.index)
    diag = {"status": "ok", "n": int(model.nobs), "events": int(use["Event_ID"].nunique()), "cities": int(use["city_clean"].nunique()), "r2": float(model.rsquared)}
    return model, se, p, diag


def run_quartile_models(full_df: pd.DataFrame, quartile_lookup: pd.DataFrame):
    need = [
        "Event_ID", "city_clean", "is_extreme", "new_hotspot_region",
        "dist_to_center_km", "ntl_avg_rad_2021", "worldpop_2020",
        "poi_log1p_2018", "poi_unique_cat_n_2018",
        "elevation_copdem_m", "slope_copdem_deg", "twi_proxy"
    ]
    x = full_df[[c for c in need if c in full_df.columns]].copy()
    x["city_clean"] = x["city_clean"].map(normalize_city)
    x = x.merge(quartile_lookup[["city_clean", "builtup_area_quartile"]], on="city_clean", how="inner")

    x["is_extreme"] = pd.to_numeric(x["is_extreme"], errors="coerce")
    x["new_hotspot_region"] = pd.to_numeric(x["new_hotspot_region"], errors="coerce").fillna(0)
    x = x[x["is_extreme"] == 1].copy()

    rename_map = {
        "twi_proxy": "twi",
        "elevation_copdem_m": "elev",
        "slope_copdem_deg": "slope",
        "ntl_avg_rad_2021": "ntl",
        "worldpop_2020": "pop",
        "dist_to_center_km": "distcenter",
        "poi_log1p_2018": "poi_log",
        "poi_unique_cat_n_2018": "poi_cat",
        "new_hotspot_region": "y_new_hotspot",
    }
    x = x.rename(columns=rename_map)

    long_rows, summary_rows = [], []
    model_specs = {
        "H1_POIlog": BASE_VARS + CENTER_ACTIVITY_VARS + [POI_LOG_VAR],
        "H2_POIcat": BASE_VARS + CENTER_ACTIVITY_VARS + [POI_CAT_VAR],
    }
    for quart in ["Q1 smallest", "Q2", "Q3", "Q4 largest"]:
        sub = x[x["builtup_area_quartile"].astype(str) == quart].copy()
        if sub.empty:
            continue
        for v in ["twi", "elev", "slope", "ntl", "pop", "distcenter", "poi_log", "poi_cat"]:
            if v in sub.columns:
                sub[v] = zscore(sub[v])

        for spec_name, rhs in model_specs.items():
            rhs = [v for v in rhs if v in sub.columns]
            model, se, p, diag = fit_event_fe_lpm_citycluster(sub, "y_new_hotspot", rhs)
            if model is None:
                continue
            for term in rhs:
                dm_term = term + "_dm"
                coef = float(model.params.get(dm_term, np.nan))
                p_val = float(p.get(dm_term, np.nan))
                long_rows.append({
                    "builtup_area_quartile": quart,
                    "model": spec_name,
                    "term": term,
                    "coef": coef,
                    "se": float(se.get(dm_term, np.nan)),
                    "p": p_val,
                    "sign_star": sign_star(coef, p_val),
                    **diag,
                })
            summary = {"builtup_area_quartile": quart, "model": spec_name, "n": diag["n"], "events": diag["events"], "cities": diag["cities"], "distcenter": "", "ntl": "", "pop": "", "poi": ""}
            for key_term in ["distcenter", "ntl", "pop"]:
                dm = key_term + "_dm"
                if dm in model.params.index:
                    summary[key_term] = sign_star(float(model.params[dm]), float(p.get(dm, np.nan)))
            poi_term = "poi_log" if spec_name == "H1_POIlog" else "poi_cat"
            dm = poi_term + "_dm"
            if dm in model.params.index:
                summary["poi"] = sign_star(float(model.params[dm]), float(p.get(dm, np.nan)))
            summary_rows.append(summary)
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def run_event_interaction(event_df: pd.DataFrame, city_panel: pd.DataFrame) -> pd.DataFrame:
    x = event_df.merge(city_panel[["city_clean", "builtup_area_km2", "builtup_area_quartile"]], on="city_clean", how="left")
    x["peak_rain_z"] = zscore(x["peak_rain"])
    x["log_event_burden_z"] = zscore(np.log1p(pd.to_numeric(x["event_burden"], errors="coerce")))
    x["builtup_area_z"] = zscore(x["builtup_area_km2"])
    x["interaction_peak_x_builtup"] = x["peak_rain_z"] * x["builtup_area_z"]

    core_terms = ["peak_rain_z", "log_event_burden_z", "builtup_area_z", "interaction_peak_x_builtup"]
    usable_terms = [t for t in core_terms if t in x.columns and x[t].notna().sum() > 0 and x[t].nunique(dropna=True) > 1]
    if len(usable_terms) < 2:
        return pd.DataFrame(columns=["term", "coef", "se", "p", "lo95", "hi95", "stars", "n", "cities", "r2"])

    use = x[["event_new_hotspot_share", "city_clean"] + usable_terms].dropna().copy()
    if use.empty or use["city_clean"].nunique() < 2:
        return pd.DataFrame(columns=["term", "coef", "se", "p", "lo95", "hi95", "stars", "n", "cities", "r2"])

    dummies = pd.get_dummies(use["city_clean"], prefix="city", drop_first=True).astype(float)
    X = pd.concat([use[usable_terms], dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = use["event_new_hotspot_share"].astype(float)
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": use["city_clean"]})
    conf = model.conf_int()
    rows = []
    for term in usable_terms:
        p = float(model.pvalues[term])
        rows.append({
            "term": term,
            "coef": float(model.params[term]),
            "se": float(model.bse[term]),
            "p": p,
            "lo95": float(conf.loc[term, 0]),
            "hi95": float(conf.loc[term, 1]),
            "stars": stars(p),
            "n": int(model.nobs),
            "cities": int(use["city_clean"].nunique()),
            "r2": float(model.rsquared),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=str, default=str(DEFAULT_BASE))
    parser.add_argument("--full-csv", type=str, default=str(DEFAULT_FULL))
    parser.add_argument("--out-dir", type=str, default="outputs_reframed_heterogeneity")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = resolve_input_path(args.base_csv, DEFAULT_BASE.name)
    full_path = resolve_input_path(args.full_csv, DEFAULT_FULL.name)

    print(f"Using base CSV: {base_path}")
    print(f"Using full CSV: {full_path}")

    base_df = pd.read_csv(base_path, low_memory=False)
    full_df = pd.read_csv(full_path, low_memory=False)

    city_panel = city_builtup_area(base_df)
    event_df, city_avg = build_event_city_tables(full_df)
    city_panel = city_panel.merge(city_avg, on="city_clean", how="left")

    city_panel_csv = out_dir / "city_heterogeneity_panel.csv"
    city_panel.to_csv(city_panel_csv, index=False, encoding="utf-8-sig")
    boxplot_png = out_dir / "city_avg_new_hotspot_share_by_builtup_quartile.png"
    make_boxplot(city_panel, boxplot_png)

    quartile_lookup = city_panel[["city_clean", "builtup_area_quartile"]].drop_duplicates()
    long_df, summary_df = run_quartile_models(full_df, quartile_lookup)
    long_csv = out_dir / "heterogeneity_quartile_results_long.csv"
    summary_csv = out_dir / "heterogeneity_quartile_stability_table.csv"
    long_df.to_csv(long_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    interaction_df = run_event_interaction(event_df, city_panel)
    interaction_csv = out_dir / "heterogeneity_event_interaction_results.csv"
    interaction_df.to_csv(interaction_csv, index=False, encoding="utf-8-sig")

    meta = {
        "base_input": str(args.base_csv),
        "full_input": str(args.full_csv),
        "n_cities_city_panel": int(city_panel["city_clean"].nunique()),
        "n_extreme_events": int(event_df["Event_ID"].nunique()),
        "quartile_groups": city_panel["builtup_area_quartile"].astype(str).value_counts(dropna=False).to_dict(),
    }
    meta_json = out_dir / "city_heterogeneity_meta.json"
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {city_panel_csv}")
    print(f"Saved: {boxplot_png}")
    print(f"Saved: {long_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {interaction_csv}")
    print(f"Saved: {meta_json}")


if __name__ == "__main__":
    main()
