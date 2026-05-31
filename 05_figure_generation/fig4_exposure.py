#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-data Fig. 4 for the coverage-gap mainline.

Figure role:
    Show that exposure outside historical hotspot inventories is large,
    concentrated across cities, robust to analytical definitions, and linked
    to city/event context.

Interpretation boundary:
    The event footprint is a reported / socially sensed flood-impact footprint,
    not a hydrodynamic inundation polygon.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
COVERAGE_DIR = REPO_ROOT / "scripts" / "coverage_gap_analysis"
FIG3_DIR = SCRIPT_DIR.parent / "fig3"

for path in [COVERAGE_DIR, FIG3_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_coverage_gap_analysis as cg


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 9.6
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 8.7
plt.rcParams["xtick.labelsize"] = 7.8
plt.rcParams["ytick.labelsize"] = 7.8
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


COLORS = {
    "population": "#C97984",
    "road": "#5A8795",
    "function": "#7EA47C",
    "inside": "#8B1E2D",
    "outside": "#00A7A7",
    "robust": "#1B8A8F",
    "robust_light": "#D9F0EF",
    "context": "#00A7A7",
    "grid": "#D9D9D9",
    "diag": "#8A8A8A",
    "text": "#222222",
    "gray": "#666666",
    "panel_bg": "#FFFFFF",
    "band": "#F5F5F5",
}

EXPOSURES = [
    {
        "key": "population",
        "label": "Population exposure",
        "short": "Population",
        "total_col": "population_exposure_total",
        "outside_col": "population_exposure_uncovered",
        "share_col": "population_exposure_outside_historical_hotspots",
        "color": COLORS["population"],
    },
    {
        "key": "road",
        "label": "Road exposure",
        "short": "Road",
        "total_col": "road_exposure_total",
        "outside_col": "road_exposure_uncovered",
        "share_col": "road_exposure_outside_historical_hotspots",
        "color": COLORS["road"],
    },
    {
        "key": "function",
        "label": "Function exposure",
        "short": "Function",
        "total_col": "function_exposure_total",
        "outside_col": "function_exposure_uncovered",
        "share_col": "function_exposure_outside_historical_hotspots",
        "color": COLORS["function"],
    },
]


PANEL_C_VARS = [
    ("z_log1p_city_population", "Urban\npopulation"),
    ("z_log1p_city_builtup_area_km2", "Built-up\narea"),
    ("z_log1p_city_road_per_grid", "Road\ndensity"),
    ("z_log1p_city_poi_per_grid", "POI\ndensity"),
    ("z_city_ntl_mean", "Night-time\nlights"),
    ("z_historical_inventory_density", "Historical\ninventory density"),
    ("z_log1p_n_historical_events", "Historical\nevent count"),
    ("z_Event_Peak_Rain", "Peak\nrainfall"),
]


def progress(msg: str) -> None:
    print(f"[fig4-coverage] {msg}", flush=True)


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def safe_div(numer: float | pd.Series, denom: float | pd.Series) -> float | np.ndarray:
    numer_arr = np.asarray(numer, dtype=float)
    denom_arr = np.asarray(denom, dtype=float)
    out = np.full_like(numer_arr, np.nan, dtype=float)
    mask = denom_arr > 0
    out[mask] = numer_arr[mask] / denom_arr[mask]
    if out.ndim == 0:
        return float(out)
    return out


def zscore(s: pd.Series) -> pd.Series:
    x = safe_numeric(s)
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=s.index)
    return (x - float(x.mean())) / sd


def iqr_summary(values: pd.Series) -> dict[str, float]:
    vals = safe_numeric(values).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return {"n": 0, "q25": np.nan, "median": np.nan, "q75": np.nan}
    return {
        "n": int(len(vals)),
        "q25": float(vals.quantile(0.25)),
        "median": float(vals.median()),
        "q75": float(vals.quantile(0.75)),
    }


def lorenz_curve(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    v = np.clip(v, 0, None)
    if len(v) == 0 or np.nansum(v) <= 0:
        return np.array([0, 1]), np.array([0, 1])
    v = np.sort(v)
    y = np.insert(np.cumsum(v) / np.sum(v), 0, 0.0)
    x = np.linspace(0, 1, len(y))
    return x, y


def gini_coefficient(values: pd.Series | np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = np.clip(v[np.isfinite(v)], 0, None)
    if len(v) == 0 or np.sum(v) <= 0:
        return np.nan
    v = np.sort(v)
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def load_or_build_primary_abs(
    cache_csv: Path,
    full_table_csv: str | Path,
    road_base_csv: str | Path,
    recompute: bool = False,
) -> pd.DataFrame:
    if cache_csv.exists() and not recompute:
        progress(f"loading cached primary absolute exposure table: {cache_csv}")
        out = read_csv(cache_csv)
        out["Event_ID"] = out["Event_ID"].astype(int)
        return out

    progress("rebuilding primary absolute exposure table from grid-event data")
    road_lookup, _road_meta = cg.load_road_lookup(Path(road_base_csv))
    full = cg.load_full_table(Path(full_table_csv))
    full, _merge_meta = cg.attach_road(full, road_lookup)
    all_events, _map = cg.run_scale_analysis(full, cg.PRIMARY_SCALE_M, keep_grid_for_map=False)
    out = all_events[
        (all_events["analysis_scale_m"] == cg.PRIMARY_SCALE_M)
        & (np.isclose(all_events["hotspot_top_share"], cg.PRIMARY_HOTSPOT_SHARE))
        & (all_events["event_impact_definition"] == cg.PRIMARY_IMPACT_DEF)
    ].copy()
    out = out.sort_values(["city_std", "Event_ID"]).reset_index(drop=True)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_csv, index=False, encoding="utf-8-sig")
    progress(f"wrote cache: {cache_csv} ({len(out)} rows)")
    return out


def build_city_exposure(primary_abs: pd.DataFrame) -> pd.DataFrame:
    agg_map = {
        "Event_ID": "nunique",
        "total_affected_grids": "sum",
        "total_reports": "sum",
    }
    for spec in EXPOSURES:
        agg_map[spec["total_col"]] = "sum"
        agg_map[spec["outside_col"]] = "sum"

    city = (
        primary_abs.groupby(["city_std", "city_clean"], as_index=False)
        .agg(agg_map)
        .rename(columns={"Event_ID": "n_extreme_events"})
    )
    for spec in EXPOSURES:
        city[f"{spec['key']}_outside_share"] = safe_div(city[spec["outside_col"]], city[spec["total_col"]])
    return city


def build_robustness_table(
    sensitivity_csv: str | Path,
    primary_event_csv: str | Path,
) -> pd.DataFrame:
    sensitivity = read_csv(sensitivity_csv)
    primary = read_csv(primary_event_csv)

    rows: list[dict[str, object]] = []

    def add_row(panel: str, label: str, source: str, values: pd.Series | None = None, row: pd.Series | None = None) -> None:
        if row is not None:
            rows.append(
                {
                    "panel": panel,
                    "definition": label,
                    "source": source,
                    "n": int(row.get("event_count", np.nan)) if np.isfinite(row.get("event_count", np.nan)) else np.nan,
                    "median": float(row["median_uncovered_impact_share"]),
                    "q25": float(row["q25_uncovered_impact_share"]),
                    "q75": float(row["q75_uncovered_impact_share"]),
                }
            )
        elif values is not None:
            s = iqr_summary(values)
            rows.append(
                {
                    "panel": panel,
                    "definition": label,
                    "source": source,
                    "n": s["n"],
                    "median": s["median"],
                    "q25": s["q25"],
                    "q75": s["q75"],
                }
            )

    top_rows = sensitivity[
        (sensitivity["analysis_scale_m"] == 1000)
        & (sensitivity["event_impact_definition"] == "gt0")
    ].copy()
    for share, label in [(0.10, "Top 10%"), (0.20, "Top 20%"), (0.30, "Top 30%")]:
        hit = top_rows[np.isclose(top_rows["hotspot_top_share"], share)]
        if not hit.empty:
            add_row("Inventory threshold", label, "sensitivity_coverage_gap.csv", row=hit.iloc[0])

    impact_rows = sensitivity[
        (sensitivity["analysis_scale_m"] == 1000)
        & (np.isclose(sensitivity["hotspot_top_share"], 0.20))
    ].copy()
    impact_labels = {
        "gt0": "flood > 0",
        "ge2": "flood >= 2",
        "top50": "event top 50%",
        "top75": "event top 75%",
    }
    for key in ["gt0", "ge2", "top50", "top75"]:
        hit = impact_rows[impact_rows["event_impact_definition"] == key]
        if not hit.empty:
            add_row("Event footprint", impact_labels[key], "sensitivity_coverage_gap.csv", row=hit.iloc[0])

    add_row("Scale / low-sample", "1 km baseline", "event_level_coverage_gap.csv", values=primary["uncovered_impact_share"])
    scale2 = sensitivity[
        (sensitivity["analysis_scale_m"] == 2000)
        & (np.isclose(sensitivity["hotspot_top_share"], 0.20))
        & (sensitivity["event_impact_definition"] == "gt0")
    ]
    if not scale2.empty:
        add_row("Scale / low-sample", "2 km aggregation", "sensitivity_coverage_gap.csv", row=scale2.iloc[0])
    add_row(
        "Scale / low-sample",
        "reports >= 20",
        "event_level_coverage_gap.csv",
        values=primary.loc[safe_numeric(primary["total_reports"]) >= 20, "uncovered_impact_share"],
    )
    add_row(
        "Scale / low-sample",
        "affected >= 10",
        "event_level_coverage_gap.csv",
        values=primary.loc[safe_numeric(primary["total_affected_grids"]) >= 10, "uncovered_impact_share"],
    )

    out = pd.DataFrame(rows)
    out["median_pct"] = out["median"] * 100
    out["q25_pct"] = out["q25"] * 100
    out["q75_pct"] = out["q75"] * 100
    return out


def build_context_table(
    primary_abs: pd.DataFrame,
    full_table_csv: str | Path,
    road_base_csv: str | Path,
) -> pd.DataFrame:
    progress("building city-event context table for panel c")
    usecols = [
        "Event_ID",
        "city_clean",
        "grid_id",
        "cell_size_m",
        "gaia_value",
        "ntl_avg_rad_2021",
        "worldpop_2020",
        "poi_count_2018",
    ]
    full = read_csv(full_table_csv, usecols=usecols)
    full["city_clean"] = full["city_clean"].astype(str).str.strip()
    full["city_std"] = full["city_clean"].map(cg.clean_city_name)
    full["grid_id"] = full["grid_id"].astype(str).str.strip()
    for col in ["cell_size_m", "gaia_value", "ntl_avg_rad_2021", "worldpop_2020", "poi_count_2018"]:
        full[col] = safe_numeric(full[col])

    road_lookup, _road_meta = cg.load_road_lookup(Path(road_base_csv))
    grid = full.sort_values(["city_std", "grid_id", "Event_ID"]).drop_duplicates(["city_std", "grid_id"]).copy()
    grid = grid.merge(road_lookup, on=["city_std", "grid_id"], how="left")
    grid["secondary_road"] = safe_numeric(grid["secondary_road"]).fillna(0)
    grid["grid_area_km2"] = (safe_numeric(grid["cell_size_m"]).fillna(1000) / 1000.0) ** 2
    grid["builtup_area_km2"] = np.where(safe_numeric(grid["gaia_value"]).fillna(0) > 0, grid["grid_area_km2"], 0.0)

    city_context = (
        grid.groupby("city_std", as_index=False)
        .agg(
            city_clean_mode=("city_clean", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            city_grid_count=("grid_id", "nunique"),
            city_population=("worldpop_2020", "sum"),
            city_builtup_area_km2=("builtup_area_km2", "sum"),
            city_poi=("poi_count_2018", "sum"),
            city_road_km=("secondary_road", "sum"),
            city_ntl_mean=("ntl_avg_rad_2021", "mean"),
        )
    )
    city_events = (
        full[["city_std", "Event_ID"]]
        .drop_duplicates()
        .groupby("city_std", as_index=False)
        .agg(city_total_events=("Event_ID", "nunique"))
    )
    city_context = city_context.merge(city_events, on="city_std", how="left")
    city_context["city_road_per_grid"] = safe_div(city_context["city_road_km"], city_context["city_grid_count"])
    city_context["city_poi_per_grid"] = safe_div(city_context["city_poi"], city_context["city_grid_count"])

    features = primary_abs.merge(city_context, on="city_std", how="left", suffixes=("", "_ctx"))
    features["n_historical_events"] = (safe_numeric(features["city_total_events"]) - 1).clip(lower=0)
    features["historical_inventory_density"] = safe_div(
        safe_numeric(features["historical_hotspot_grids"]),
        safe_numeric(features["city_grid_count"]),
    )

    features["log1p_city_population"] = np.log1p(safe_numeric(features["city_population"]).clip(lower=0))
    features["log1p_city_builtup_area_km2"] = np.log1p(safe_numeric(features["city_builtup_area_km2"]).clip(lower=0))
    features["log1p_city_road_per_grid"] = np.log1p(safe_numeric(features["city_road_per_grid"]).clip(lower=0))
    features["log1p_city_poi_per_grid"] = np.log1p(safe_numeric(features["city_poi_per_grid"]).clip(lower=0))
    features["log1p_n_historical_events"] = np.log1p(safe_numeric(features["n_historical_events"]).clip(lower=0))
    features["log1p_function_exposure_uncovered"] = np.log1p(
        safe_numeric(features["function_exposure_uncovered"]).clip(lower=0)
    )

    for col in [
        "log1p_city_population",
        "log1p_city_builtup_area_km2",
        "log1p_city_road_per_grid",
        "log1p_city_poi_per_grid",
        "city_ntl_mean",
        "historical_inventory_density",
        "log1p_n_historical_events",
        "Event_Peak_Rain",
        "log1p_function_exposure_uncovered",
    ]:
        features[f"z_{col}"] = zscore(features[col])

    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def style_axis(ax: plt.Axes, ygrid: bool = True, xgrid: bool = False) -> None:
    ax.set_facecolor(COLORS["panel_bg"])
    if ygrid:
        ax.grid(True, axis="y", linestyle="--", linewidth=0.65, color=COLORS["grid"], alpha=0.75)
    else:
        ax.grid(False, axis="y")
    if xgrid:
        ax.grid(True, axis="x", linestyle="--", linewidth=0.55, color=COLORS["grid"], alpha=0.45)
    else:
        ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")
    ax.tick_params(width=0.9, length=3.5, color="#333333", pad=2)


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.2,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )


def draw_lorenz_panel(
    ax: plt.Axes,
    city_exposure: pd.DataFrame,
    spec: dict,
    letter: str | None = None,
    show_ylabel: bool = False,
    show_xlabel: bool = False,
) -> None:
    style_axis(ax, ygrid=True, xgrid=False)
    values = safe_numeric(city_exposure[spec["outside_col"]]).clip(lower=0)
    x, y = lorenz_curve(values)
    gini = gini_coefficient(values)

    ax.plot([0, 1], [0, 1], color=COLORS["diag"], linestyle="--", linewidth=0.9, zorder=1)
    ax.plot(x, y, color=spec["color"], linewidth=2.1, zorder=2)
    ax.fill_between(x, y, 0, color=spec["color"], alpha=0.08, zorder=0)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title(spec["label"], loc="left", pad=4)
    if show_xlabel:
        ax.set_xlabel("Cumulative share of cities")
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("Cumulative share of\noutside exposure")
    else:
        ax.set_yticklabels([])
    ax.text(
        0.97,
        0.08,
        f"Gini {gini:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#BDBDBD", linewidth=0.45, alpha=0.92, pad=1.3),
    )
    if letter:
        panel_letter(ax, letter)


def draw_robustness_panel(
    ax: plt.Axes,
    robust: pd.DataFrame,
    panel: str,
    title: str,
    letter: str | None = None,
    show_ylabel: bool = False,
    show_note: bool = False,
) -> None:
    style_axis(ax, ygrid=True, xgrid=False)
    sub = robust[robust["panel"] == panel].copy().reset_index(drop=True)
    x = np.arange(len(sub))
    med = sub["median_pct"].to_numpy(float)
    q25 = sub["q25_pct"].to_numpy(float)
    q75 = sub["q75_pct"].to_numpy(float)

    ax.vlines(x, q25, q75, color=COLORS["robust"], linewidth=1.8, alpha=0.70, zorder=2)
    ax.scatter(x, med, s=31, color=COLORS["robust"], edgecolor="white", linewidth=0.35, zorder=3)
    for xi, yi in zip(x, med):
        ax.text(
            xi,
            min(yi + 4.0, 98),
            f"{yi:.0f}",
            ha="center",
            va="bottom",
            fontsize=6.1,
            fontweight="bold",
            color=COLORS["text"],
        )
    ax.axhline(50, color="#AAAAAA", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_title(title, loc="left", pad=4)
    ax.set_ylim(35, 100)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_xlim(-0.55, max(len(sub) - 0.45, 0.5))
    ax.set_xticks(x)
    ax.set_xticklabels(sub["definition"], rotation=40, ha="right", fontsize=6.7)
    if show_ylabel:
        ax.set_ylabel("Median coverage gap (%)")
    else:
        ax.set_yticklabels([])
    if show_note:
        ax.text(
            0.97,
            0.04,
            "point: median\nline: IQR",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.9,
            color=COLORS["gray"],
        )
    if letter:
        panel_letter(ax, letter)


def binned_median(x: np.ndarray, y: np.ndarray, n_bins: int = 9) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 25:
        return np.array([]), np.array([])
    bins = np.unique(np.nanquantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(bins) < 3:
        return np.array([]), np.array([])
    centers, medians = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        hit = (x >= lo) & (x <= hi)
        if hit.sum() >= 8:
            centers.append(float(np.nanmedian(x[hit])))
            medians.append(float(np.nanmedian(y[hit])))
    return np.asarray(centers), np.asarray(medians)


def draw_context_panel(
    ax: plt.Axes,
    context: pd.DataFrame,
    x_col: str,
    title: str,
    letter: str | None = None,
    show_ylabel: bool = False,
    show_xlabel: bool = False,
) -> None:
    style_axis(ax, ygrid=True, xgrid=True)
    y_col = "z_log1p_function_exposure_uncovered"
    sub = context[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    x = sub[x_col].to_numpy(float)
    y = sub[y_col].to_numpy(float)
    ax.scatter(
        np.clip(x, -3.1, 3.1),
        np.clip(y, -2.6, 3.2),
        s=8.5,
        color=COLORS["context"],
        alpha=0.20,
        edgecolors="none",
        zorder=2,
    )
    cx, cy = binned_median(x, y)
    if len(cx) > 1:
        ax.plot(np.clip(cx, -3.1, 3.1), np.clip(cy, -2.6, 3.2), color="#111111", linewidth=1.45, zorder=3)
    rho = pd.Series(x).corr(pd.Series(y), method="spearman") if len(sub) > 3 else np.nan
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-2.4, 3.05)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_title(title, loc="left", pad=4)
    ax.text(
        0.04,
        0.91,
        f"rho={rho:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.8),
    )
    if show_ylabel:
        ax.set_ylabel("Std. log(function\noutside + 1)")
    else:
        ax.set_yticklabels([])
    if show_xlabel:
        ax.set_xlabel("Standardized attribute")
    else:
        ax.set_xticklabels([])
    if letter:
        panel_letter(ax, letter, x=-0.32, y=1.10)


def plot_fig4(
    output_dir: str | Path = REPO_ROOT / "outputs" / "figures",
    output_stem: str = "Figure4_coverage_gap_exposure_real",
    full_table_csv: str | Path = cg.DEFAULT_FULL_TABLE,
    road_base_csv: str | Path = cg.DEFAULT_ROAD_BASE,
    event_summary_csv: str | Path = COVERAGE_DIR / "outputs" / "event_level_coverage_gap.csv",
    sensitivity_csv: str | Path = COVERAGE_DIR / "outputs" / "sensitivity_coverage_gap.csv",
    recompute_abs: bool = False,
    show: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_abs_csv = output_dir / f"{output_stem}_event_abs.csv"
    city_exposure_csv = output_dir / f"{output_stem}_city_exposure.csv"
    robustness_csv = output_dir / f"{output_stem}_robustness.csv"
    context_csv = output_dir / f"{output_stem}_context_gradients.csv"

    primary_abs = load_or_build_primary_abs(event_abs_csv, full_table_csv, road_base_csv, recompute=recompute_abs)
    city_exposure = build_city_exposure(primary_abs)
    robustness = build_robustness_table(sensitivity_csv, event_summary_csv)
    context = build_context_table(primary_abs, full_table_csv, road_base_csv)

    city_exposure.to_csv(city_exposure_csv, index=False, encoding="utf-8-sig")
    robustness.to_csv(robustness_csv, index=False, encoding="utf-8-sig")
    context.to_csv(context_csv, index=False, encoding="utf-8-sig")
    progress(f"wrote figure data tables to {output_dir}")

    fig = plt.figure(figsize=(14.5, 10.3))
    fig.patch.set_facecolor("white")

    L, R, B, T = 0.055, 0.975, 0.105, 0.955

    left_x = L
    left_w = 0.235
    left_bottom = B + 0.075
    left_top = T
    left_gap = 0.050
    left_h = (left_top - left_bottom - 2 * left_gap) / 3

    right_x = 0.345
    right_w = R - right_x
    b_gap = 0.040
    b_w = (right_w - 2 * b_gap) / 3
    b_h = 0.245
    b_y = T - b_h

    c_gap_x = 0.038
    c_gap_y = 0.082
    c_w = (right_w - 3 * c_gap_x) / 4
    c_h = 0.190
    c_top = 0.570
    c_row1_y = c_top - c_h
    c_row2_y = c_row1_y - c_gap_y - c_h

    outside_share_notes = []
    for i, spec in enumerate(EXPOSURES):
        y = left_top - (i + 1) * left_h - i * left_gap
        ax = fig.add_axes([left_x, y, left_w, left_h])
        median_outside_pct = 100 * float(safe_numeric(primary_abs[spec["share_col"]]).median())
        outside_share_notes.append(f"{spec['short']} {median_outside_pct:.0f}/{100 - median_outside_pct:.0f}%")
        draw_lorenz_panel(
            ax,
            city_exposure,
            spec,
            letter="a" if i == 0 else None,
            show_ylabel=(i == 1),
            show_xlabel=(i == 2),
        )

    b_specs = [
        ("Inventory threshold", "Inventory threshold"),
        ("Event footprint", "Event footprint"),
        ("Scale / low-sample", "Scale / low-sample"),
    ]
    for i, (panel, title) in enumerate(b_specs):
        ax = fig.add_axes([right_x + i * (b_w + b_gap), b_y, b_w, b_h])
        draw_robustness_panel(
            ax,
            robustness,
            panel,
            title,
            letter="b" if i == 0 else None,
            show_ylabel=(i == 0),
            show_note=(i == 2),
        )

    fig.text(
        L,
        0.086,
        "Median event-level exposure split, outside/inside: " + ", ".join(outside_share_notes) + ".",
        ha="left",
        va="center",
        fontsize=7.4,
        color=COLORS["gray"],
    )

    for idx, (x_col, title) in enumerate(PANEL_C_VARS):
        row = idx // 4
        col = idx % 4
        ax = fig.add_axes([
            right_x + col * (c_w + c_gap_x),
            c_row1_y if row == 0 else c_row2_y,
            c_w,
            c_h,
        ])
        draw_context_panel(
            ax,
            context,
            x_col,
            title,
            letter="c" if idx == 0 else None,
            show_ylabel=(col == 0),
            show_xlabel=(row == 1),
        )

    legend_ax = fig.add_axes([L, 0.030, R - L, 0.052])
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], color=COLORS["diag"], linestyle="--", linewidth=1.0, label="Equality line"),
        Line2D([0], [0], color=COLORS["population"], linewidth=2.2, label="Population outside"),
        Line2D([0], [0], color=COLORS["road"], linewidth=2.2, label="Road outside"),
        Line2D([0], [0], color=COLORS["function"], linewidth=2.2, label="Function outside"),
        Line2D([0], [0], marker="o", color=COLORS["robust"], linestyle="None", markersize=5.2, label="Median"),
        Line2D([0], [0], color="#111111", linewidth=1.4, label="Binned median"),
    ]
    legend_ax.legend(
        handles=handles,
        loc="center left",
        ncol=6,
        frameon=False,
        fontsize=7.5,
        handlelength=1.55,
        columnspacing=0.62,
        handletextpad=0.38,
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
    plot_fig4(show=False)

