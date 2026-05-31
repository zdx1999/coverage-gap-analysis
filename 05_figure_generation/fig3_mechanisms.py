#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-data Fig. 3 for the coverage-gap mainline.

Figure role:
    Show which historical-record, urban-exposure, physical-setting, and
    event/reporting variables are associated with larger event-time reported
    impacts outside historical hotspot inventories.

Interpretation boundary:
    The outcome is a coverage gap in a public-reporting / socially sensed
    flood-impact footprint, not a hydrodynamic inundation boundary.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
COVERAGE_DIR = REPO_ROOT / "scripts" / "coverage_gap_analysis"

for path in [COVERAGE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 9.8
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 8.8
plt.rcParams["xtick.labelsize"] = 7.8
plt.rcParams["ytick.labelsize"] = 6.8
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


COLORS = {
    "record": "#8B1E2D",
    "urban": "#1B8A8F",
    "physical": "#6A4C93",
    "event": "#C9911A",
    "weak": "#BDBDBD",
    "grid": "#D9D9D9",
    "band": "#F4F4F4",
    "text": "#222222",
    "gray": "#666666",
}

FAMILY_LABELS = {
    "record": "Historical-record structure",
    "urban": "Urban activity and exposure",
    "physical": "Physical setting",
    "event": "Event intensity / reporting",
}

FAMILY_ORDER = {"record": 0, "urban": 1, "physical": 2, "event": 3}


@dataclass(frozen=True)
class Predictor:
    column: str
    label: str
    family: str


PREDICTORS = [
    Predictor("historical_inventory_density", "Historical inventory density", "record"),
    Predictor("log1p_historical_report_density", "Historical report density", "record"),
    Predictor("log1p_n_historical_events", "Historical event count", "record"),
    Predictor("low_history_city", "Low-history city", "record"),
    Predictor("log1p_affected_pop_per_grid", "Population density", "urban"),
    Predictor("log1p_affected_poi_per_grid", "POI intensity", "urban"),
    Predictor("affected_ntl_mean", "Night-time lights", "urban"),
    Predictor("log1p_affected_road_per_grid", "Road density", "urban"),
    Predictor("affected_dist_center_mean", "Distance to city centre", "urban"),
    Predictor("affected_slope_mean", "Mean slope", "physical"),
    Predictor("affected_twi_mean", "TWI proxy", "physical"),
    Predictor("Event_Peak_Rain", "Event peak rainfall", "event"),
    Predictor("log1p_total_reports", "Total reports", "event"),
    Predictor("log1p_total_affected_grids", "Affected footprint size", "event"),
]

EVENT_CONTROLS = ["Event_Peak_Rain", "log1p_total_reports", "log1p_total_affected_grids"]
CITY_CONTEXT_CONTROLS = ["log1p_city_population", "log1p_city_grids", "low_history_city"]


def progress(msg: str) -> None:
    print(f"[fig3-coverage] {msg}", flush=True)


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def zscore(s: pd.Series) -> pd.Series | None:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return None
    return (x - float(x.mean())) / sd


def build_feature_table(
    event_csv: str | Path,
    full_grid_csv: str | Path,
    road_csv: str | Path,
    split_half_csv: str | Path,
) -> pd.DataFrame:
    progress("loading coverage-gap event table")
    event = read_csv(event_csv)
    event["Event_ID"] = event["Event_ID"].astype(int)

    full_cols = [
        "Event_ID",
        "city_clean",
        "grid_id",
        "dist_to_center_km",
        "gaia_value",
        "ntl_avg_rad_2021",
        "worldpop_2020",
        "elevation_copdem_m",
        "slope_copdem_deg",
        "twi_proxy",
        "flood_count",
        "poi_count_2018",
    ]
    road_cols = ["city_clean", "grid_id", "road_len_km_grip4_l2", "road_len_km_grip4_l3"]

    progress("loading grid-event table and road exposure")
    full = read_csv(full_grid_csv, usecols=full_cols)
    full["Event_ID"] = full["Event_ID"].astype(int)
    road = read_csv(road_csv, usecols=road_cols)
    full = full.merge(road, on=["city_clean", "grid_id"], how="left")
    for col in ["road_len_km_grip4_l2", "road_len_km_grip4_l3"]:
        full[col] = pd.to_numeric(full[col], errors="coerce").fillna(0.0)
    full["secondary_road_km"] = full["road_len_km_grip4_l2"] + full["road_len_km_grip4_l3"]

    numeric_cols = [
        "dist_to_center_km",
        "gaia_value",
        "ntl_avg_rad_2021",
        "worldpop_2020",
        "elevation_copdem_m",
        "slope_copdem_deg",
        "twi_proxy",
        "flood_count",
        "poi_count_2018",
        "secondary_road_km",
    ]
    for col in numeric_cols:
        full[col] = pd.to_numeric(full[col], errors="coerce")

    progress("computing city-level historical context")
    city_grid = full.sort_values(["city_clean", "grid_id", "Event_ID"]).drop_duplicates(["city_clean", "grid_id"])
    city_context = (
        city_grid.groupby("city_clean", dropna=False)
        .agg(
            city_grid_count=("grid_id", "size"),
            city_population=("worldpop_2020", "sum"),
            city_poi=("poi_count_2018", "sum"),
            city_road_km=("secondary_road_km", "sum"),
            city_ntl_mean=("ntl_avg_rad_2021", "mean"),
            city_slope_mean=("slope_copdem_deg", "mean"),
            city_slope_p75=("slope_copdem_deg", lambda s: float(np.nanpercentile(s, 75))),
            city_twi_mean=("twi_proxy", "mean"),
        )
        .reset_index()
    )

    city_reports = (
        full.groupby("city_clean", dropna=False)["flood_count"]
        .sum(min_count=1)
        .rename("city_total_reports_all_events")
        .reset_index()
    )
    city_events = (
        full[["city_clean", "Event_ID"]]
        .drop_duplicates()
        .groupby("city_clean", dropna=False)
        .size()
        .rename("city_total_events")
        .reset_index()
    )
    city_context = city_context.merge(city_reports, on="city_clean", how="left").merge(city_events, on="city_clean", how="left")

    progress("aggregating event-time reported impact footprints")
    event_keys = event[["city_clean", "Event_ID"]].drop_duplicates()
    sub = full.merge(event_keys, on=["city_clean", "Event_ID"], how="inner")
    affected = sub[pd.to_numeric(sub["flood_count"], errors="coerce").fillna(0) > 0].copy()
    keys = ["city_clean", "Event_ID"]
    event_agg = (
        affected.groupby(keys, dropna=False)
        .agg(
            affected_pop_total=("worldpop_2020", "sum"),
            affected_poi_total=("poi_count_2018", "sum"),
            affected_road_total=("secondary_road_km", "sum"),
            affected_ntl_mean=("ntl_avg_rad_2021", "mean"),
            affected_dist_center_mean=("dist_to_center_km", "mean"),
            affected_elevation_mean=("elevation_copdem_m", "mean"),
            affected_slope_mean=("slope_copdem_deg", "mean"),
            affected_twi_mean=("twi_proxy", "mean"),
            affected_builtup_mean=("gaia_value", "mean"),
        )
        .reset_index()
    )

    feat = event.merge(city_context, on="city_clean", how="left").merge(event_agg, on=keys, how="left")

    if Path(split_half_csv).exists():
        split = read_csv(split_half_csv, usecols=["city_clean", "low_history_city", "median_jaccard", "median_overlap_coefficient"])
        feat = feat.merge(split, on="city_clean", how="left")
    else:
        feat["low_history_city"] = False
        feat["median_jaccard"] = np.nan
        feat["median_overlap_coefficient"] = np.nan

    feat["low_history_city"] = feat["low_history_city"].fillna(False).astype(float)
    feat["n_historical_events"] = (pd.to_numeric(feat["city_total_events"], errors="coerce") - 1).clip(lower=0)
    feat["historical_inventory_density"] = (
        pd.to_numeric(feat["historical_hotspot_grids"], errors="coerce")
        / pd.to_numeric(feat["city_grid_count"], errors="coerce")
    )
    hist_reports = (
        pd.to_numeric(feat["city_total_reports_all_events"], errors="coerce")
        - pd.to_numeric(feat["total_reports"], errors="coerce")
    ).clip(lower=0)
    feat["historical_report_density"] = hist_reports / pd.to_numeric(feat["city_grid_count"], errors="coerce")

    feat["affected_pop_per_grid"] = feat["affected_pop_total"] / pd.to_numeric(feat["total_affected_grids"], errors="coerce")
    feat["affected_poi_per_grid"] = feat["affected_poi_total"] / pd.to_numeric(feat["total_affected_grids"], errors="coerce")
    feat["affected_road_per_grid"] = feat["affected_road_total"] / pd.to_numeric(feat["total_affected_grids"], errors="coerce")

    for src, dst in [
        ("historical_report_density", "log1p_historical_report_density"),
        ("n_historical_events", "log1p_n_historical_events"),
        ("city_population", "log1p_city_population"),
        ("city_grid_count", "log1p_city_grids"),
        ("total_reports", "log1p_total_reports"),
        ("total_affected_grids", "log1p_total_affected_grids"),
        ("affected_pop_per_grid", "log1p_affected_pop_per_grid"),
        ("affected_poi_per_grid", "log1p_affected_poi_per_grid"),
        ("affected_road_per_grid", "log1p_affected_road_per_grid"),
    ]:
        feat[dst] = np.log1p(pd.to_numeric(feat[src], errors="coerce").clip(lower=0))

    city_for_groups = feat.drop_duplicates("city_clean").copy()
    pop_q1, pop_q2 = city_for_groups["city_population"].quantile([1 / 3, 2 / 3])
    terrain_q2 = city_for_groups["city_slope_p75"].quantile(2 / 3)
    feat["city_size_group"] = np.where(
        feat["city_population"] >= pop_q2,
        "mega_large",
        np.where(feat["city_population"] >= pop_q1, "medium", "small"),
    )
    feat["terrain_constrained_city"] = feat["city_slope_p75"] >= terrain_q2

    keep = ["city_clean", "city_std", "Event_ID", "uncovered_impact_share"] + [p.column for p in PREDICTORS]
    missing = [c for c in keep if c not in feat.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return feat


def fit_focal_effect(
    df: pd.DataFrame,
    outcome: str,
    focal: str,
    controls: list[str],
    model_type: str,
) -> dict[str, float | str] | None:
    controls = [c for c in controls if c != focal and c in df.columns]
    cols = [outcome, "city_clean", focal] + controls
    use = df[cols].copy()
    for col in [outcome, focal] + controls:
        use[col] = pd.to_numeric(use[col], errors="coerce")
    use = use.replace([np.inf, -np.inf], np.nan).dropna()

    if len(use) < 30 or use["city_clean"].nunique() < 8:
        return None

    y = zscore(use[outcome])
    x_focal = zscore(use[focal])
    if y is None or x_focal is None:
        return None

    X = pd.DataFrame({focal: x_focal}, index=use.index)
    kept_controls = []
    for control in controls:
        z = zscore(use[control])
        if z is not None:
            X[control] = z
            kept_controls.append(control)
    X = sm.add_constant(X, has_constant="add")

    try:
        groups = use["city_clean"].astype(str)
        if groups.nunique() >= 8:
            model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
            cov_type = "city-clustered"
        else:
            model = sm.OLS(y, X).fit(cov_type="HC3")
            cov_type = "HC3"
    except Exception:
        model = sm.OLS(y, X).fit(cov_type="HC3")
        cov_type = "HC3"

    coef = float(model.params.get(focal, np.nan))
    se = float(model.bse.get(focal, np.nan))
    pval = float(model.pvalues.get(focal, np.nan))
    return {
        "model_type": model_type,
        "coef": coef,
        "ci_low": coef - 1.96 * se,
        "ci_high": coef + 1.96 * se,
        "se": se,
        "p": pval,
        "n_events": int(len(use)),
        "n_cities": int(use["city_clean"].nunique()),
        "r2": float(model.rsquared),
        "cov_type": cov_type,
        "controls": "+".join(kept_controls),
    }


def build_effect_table(features: pd.DataFrame) -> pd.DataFrame:
    panels = [
        ("a", "Mega / large cities", features["city_size_group"] == "mega_large", "Adjusted"),
        ("b", "Medium cities", features["city_size_group"] == "medium", "Adjusted"),
        ("c", "Terrain-constrained cities", features["terrain_constrained_city"].astype(bool), "Adjusted"),
    ]

    rows: list[dict[str, object]] = []
    outcome = "uncovered_impact_share"

    for letter, panel, mask, model_type in panels:
        sub = features[mask].copy()
        for order, pred in enumerate(PREDICTORS):
            fit = fit_focal_effect(sub, outcome, pred.column, EVENT_CONTROLS, model_type)
            if fit is None:
                continue
            rows.append(
                {
                    "panel": panel,
                    "letter": letter,
                    "variable": pred.label,
                    "column": pred.column,
                    "family": pred.family,
                    "display_order": order,
                    **fit,
                }
            )

    full_specs = [
        ("Baseline", []),
        ("Rain/report adjusted", EVENT_CONTROLS),
        ("City-context adjusted", EVENT_CONTROLS + CITY_CONTEXT_CONTROLS),
    ]
    for model_type, controls in full_specs:
        for order, pred in enumerate(PREDICTORS):
            fit = fit_focal_effect(features, outcome, pred.column, controls, model_type)
            if fit is None:
                continue
            rows.append(
                {
                    "panel": "Full national sample",
                    "letter": "d",
                    "variable": pred.label,
                    "column": pred.column,
                    "family": pred.family,
                    "display_order": order,
                    **fit,
                }
            )

    effects = pd.DataFrame(rows)
    if effects.empty:
        raise ValueError("No model effects were estimated.")
    effects["significant_05"] = pd.to_numeric(effects["p"], errors="coerce") < 0.05
    return effects


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.55, color=COLORS["grid"], alpha=0.60)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#333333")
    ax.tick_params(width=0.75, length=3.2, color="#333333")


def compact_label(label: str) -> str:
    return (
        label.replace("Historical ", "Hist. ")
        .replace("inventory", "inv.")
        .replace("density", "dens.")
        .replace("Number of ", "No. ")
        .replace("Distance to city centre", "Dist. to centre")
        .replace("Affected footprint size", "Footprint size")
        .replace("Event peak rainfall", "Peak rainfall")
    )


def marker_for_model(model_type: str) -> str:
    if model_type == "Baseline":
        return "o"
    if model_type == "Rain/report adjusted":
        return "^"
    if model_type == "City-context adjusted":
        return "s"
    return "o"


def color_for_family(family: str) -> str:
    return COLORS.get(family, COLORS["weak"])


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.11,
        1.03,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=14.5,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )


def draw_family_bands(ax: plt.Axes, df: pd.DataFrame, y: np.ndarray, xlim: tuple[float, float]) -> None:
    start = 0
    while start < len(df):
        fam = df.iloc[start]["family"]
        end = start
        while end + 1 < len(df) and df.iloc[end + 1]["family"] == fam:
            end += 1
        y_top = y[start]
        y_bottom = y[end]
        ax.add_patch(
            Rectangle(
                (xlim[0], y_bottom - 0.46),
                xlim[1] - xlim[0],
                y_top - y_bottom + 0.92,
                facecolor=COLORS["band"],
                edgecolor="none",
                alpha=0.48,
                zorder=0,
            )
        )
        start = end + 1


def draw_effect_panel(
    ax: plt.Axes,
    effects: pd.DataFrame,
    panel: str,
    letter: str,
    title: str,
    xlim: tuple[float, float],
    full_panel: bool = False,
    show_xlabel: bool = True,
) -> None:
    style_axis(ax)
    panel_letter(ax, letter)

    df = effects[effects["panel"] == panel].copy()
    if df.empty:
        ax.text(0.5, 0.5, "No estimable rows", transform=ax.transAxes, ha="center", va="center")
        return

    df["family_order"] = df["family"].map(FAMILY_ORDER)
    df["model_order"] = df["model_type"].map({"Baseline": 0, "Rain/report adjusted": 1, "City-context adjusted": 2, "Adjusted": 0}).fillna(0)
    if full_panel:
        df = df.sort_values(["family_order", "display_order", "model_order"]).reset_index(drop=True)
        if df["model_type"].nunique() == 1:
            labels = [compact_label(v) for v in df["variable"]]
        else:
            labels = [
                f"{compact_label(row.variable)} - "
                f"{'base' if row.model_type == 'Baseline' else 'rain/report' if row.model_type == 'Rain/report adjusted' else 'city ctx'}"
                for row in df.itertuples()
            ]
        y_font = 5.45
    else:
        df = df.sort_values(["family_order", "display_order"]).reset_index(drop=True)
        labels = [compact_label(v) for v in df["variable"]]
        y_font = 6.55

    n = len(df)
    y = np.arange(n)[::-1]
    draw_family_bands(ax, df, y, xlim)
    ax.axvline(0, color="#111111", linewidth=0.9, zorder=1)

    for yi, row in zip(y, df.itertuples(index=False)):
        color = color_for_family(row.family)
        alpha = 0.92 if row.significant_05 else 0.55
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=color, linewidth=1.65, alpha=alpha, zorder=3)
        ax.scatter(
            row.coef,
            yi,
            s=24 if full_panel else 32,
            marker=marker_for_model(row.model_type),
            facecolor=color if row.significant_05 else "white",
            edgecolor=color,
            linewidth=1.05,
            alpha=0.96,
            zorder=4,
        )
        ax.add_patch(
            Rectangle(
                (xlim[0] + (xlim[1] - xlim[0]) * 0.008, yi - 0.34),
                (xlim[1] - xlim[0]) * 0.018,
                0.68,
                facecolor=color,
                edgecolor="white",
                linewidth=0.25,
                clip_on=True,
                zorder=5,
            )
        )

    ax.set_xlim(xlim)
    ax.set_ylim(-0.8, n - 0.2)
    tick_step = 0.25 if (xlim[1] - xlim[0]) <= 1.25 else 0.5
    ticks = np.arange(np.ceil(xlim[0] / tick_step) * tick_step, xlim[1] + 1e-9, tick_step)
    ax.set_xticks(np.round(ticks, 2))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=y_font)
    if show_xlabel:
        ax.set_xlabel("Standardized association with coverage gap", labelpad=5)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_title(title, loc="left", pad=8)

    ax.text(
        xlim[0] + 0.24 * (xlim[1] - xlim[0]),
        n - 0.38,
        "smaller gap (-)",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["gray"],
    )
    ax.text(
        xlim[0] + 0.76 * (xlim[1] - xlim[0]),
        n - 0.38,
        "larger gap (+)",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["gray"],
    )


def x_limits(effects: pd.DataFrame) -> tuple[float, float]:
    vals = pd.to_numeric(effects[["ci_low", "ci_high"]].stack(), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return -0.6, 0.6
    low = float(vals.min())
    high = float(vals.max())
    span = max(high - low, 0.8)
    low = min(low - span * 0.035, -0.35)
    high = max(high + span * 0.035, 0.35)
    return max(low, -1.35), min(high, 1.20)


def plot_fig3(
    output_dir: str | Path = REPO_ROOT / "outputs" / "figures",
    output_stem: str = "Figure3_coverage_gap_mechanism_real",
    event_csv: str | Path = REPO_ROOT / "outputs" / "coverage_gap_analysis" / "event_level_coverage_gap.csv",
    full_grid_csv: str | Path = REPO_ROOT / "data" / "city_event_grid_full_gaia_v4_poi_repaired_253cities.csv",
    road_csv: str | Path = REPO_ROOT / "data" / "city_grid_base_centroids_gaia_with_gee_terrain_repaired_253cities.csv",
    split_half_csv: str | Path = REPO_ROOT / "outputs" / "validity_checks" / "historical_inventory_split_half_stability.csv",
    show: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = build_feature_table(event_csv, full_grid_csv, road_csv, split_half_csv)
    effects = build_effect_table(features)

    features.to_csv(output_dir / f"{output_stem}_model_inputs.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(output_dir / f"{output_stem}_effects.csv", index=False, encoding="utf-8-sig")

    xlim = x_limits(effects)
    fig = plt.figure(figsize=(15.8, 10.2))
    fig.patch.set_facecolor("white")

    L, R, T, B = 0.120, 0.985, 0.920, 0.150
    gap_x = 0.130
    left_w = 0.365
    right_w = R - L - left_w - gap_x
    left_x = L
    right_x = left_x + left_w + gap_x
    left_gap_y = 0.050
    left_h = (T - B - 2 * left_gap_y) / 3

    panel_specs = [
        ("Mega / large cities", "a", "Mega / large cities"),
        ("Medium cities", "b", "Medium cities"),
        ("Terrain-constrained cities", "c", "Terrain-constrained cities"),
    ]
    for i, (panel, letter, base_title) in enumerate(panel_specs):
        sub = features[
            (features["city_size_group"] == "mega_large")
            if letter == "a"
            else (features["city_size_group"] == "medium")
            if letter == "b"
            else features["terrain_constrained_city"].astype(bool)
        ]
        title = f"{base_title}  n={len(sub)}, cities={sub['city_clean'].nunique()}"
        y = T - (i + 1) * left_h - i * left_gap_y
        ax = fig.add_axes([left_x, y, left_w, left_h])
        draw_effect_panel(ax, effects, panel, letter, title, xlim, full_panel=False, show_xlabel=(letter == "c"))

    ax_d = fig.add_axes([right_x, B, right_w, T - B])
    title_d = f"Full national sample  n={len(features)}, cities={features['city_clean'].nunique()}"
    draw_effect_panel(ax_d, effects, "Full national sample", "d", title_d, xlim, full_panel=True, show_xlabel=True)

    legend_ax = fig.add_axes([L, 0.035, R - L, 0.082])
    legend_ax.axis("off")
    family_handles = [
        Patch(facecolor=COLORS[k], edgecolor="none", label=FAMILY_LABELS[k])
        for k in ["record", "urban", "physical", "event"]
    ]
    model_handles = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", linestyle="None", label="Baseline"),
        Line2D([0], [0], marker="^", color="black", markerfacecolor="white", linestyle="None", label="Rain/report adjusted"),
        Line2D([0], [0], marker="s", color="black", markerfacecolor="white", linestyle="None", label="City-context adjusted"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="black", linestyle="None", label="p < 0.05"),
    ]
    legend1 = legend_ax.legend(
        handles=family_handles,
        loc="center left",
        bbox_to_anchor=(0.00, 0.68),
        ncol=4,
        frameon=False,
        fontsize=7.9,
        columnspacing=1.05,
        handlelength=1.25,
    )
    legend_ax.add_artist(legend1)
    legend_ax.legend(
        handles=model_handles,
        loc="center left",
        bbox_to_anchor=(0.00, 0.18),
        ncol=4,
        frameon=False,
        fontsize=7.8,
        columnspacing=1.10,
        handletextpad=0.45,
    )

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    progress(f"saved PNG: {png_path.resolve()}")
    progress(f"saved PDF: {pdf_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_fig3(show=False)





