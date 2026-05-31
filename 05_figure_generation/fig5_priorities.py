#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-data Fig. 5 for the coverage-gap mainline.

Figure role:
    Translate historical-inventory coverage gaps into city-level governance
    priorities under observed extreme events and a reproducible tail-rainfall
    stress-test scenario.

Interpretation boundary:
    The stress scenario is a sensitivity / stress-test based on observed
    event peak rainfall, not a climate projection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

try:
    from pypinyin import lazy_pinyin
except Exception:
    lazy_pinyin = None


SCRIPT_DIR = Path(__file__).resolve().parent
COVERAGE_DIR = SCRIPT_DIR.parents[1] / "scripts" / "coverage_gap_analysis"
REPO_ROOT = SCRIPT_DIR.parents[1]
FIG4_DIR = SCRIPT_DIR.parent / "fig4"

for path in [COVERAGE_DIR, FIG4_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_coverage_gap_analysis as cg
import fig4_exposure as fig4


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.titlesize"] = 9.2
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 8.0
plt.rcParams["xtick.labelsize"] = 6.2
plt.rcParams["ytick.labelsize"] = 7.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


GDP_CMAP = LinearSegmentedColormap.from_list(
    "gdp_rb",
    ["#D95F5F", "#F3C2C2", "#D7E3F3", "#4F7FB7"],
)

COLORS = {
    "bar_edge": "white",
    "dot": "#111111",
    "grid": "#D9D9D9",
    "text": "#222222",
    "gray": "#666666",
    "stress": "#111111",
}

ROW_SPECS = [
    {
        "metric": "uncovered_impact_space",
        "label": "Uncovered impact\nspace",
        "unit": "km^2",
        "outside_col": "uncovered_affected_grids",
        "total_col": "total_affected_grids",
        "value_scale": 1.0,
        "axis_label": "km$^2$",
    },
    {
        "metric": "population_outside",
        "label": "Population outside\nhistorical hotspots",
        "unit": "10^3 persons",
        "outside_col": "population_exposure_uncovered",
        "total_col": "population_exposure_total",
        "value_scale": 1000.0,
        "axis_label": "10$^3$ persons",
    },
    {
        "metric": "road_outside",
        "label": "Road outside\nhistorical hotspots",
        "unit": "km",
        "outside_col": "road_exposure_uncovered",
        "total_col": "road_exposure_total",
        "value_scale": 1.0,
        "axis_label": "km",
    },
    {
        "metric": "function_outside",
        "label": "Function outside\nhistorical hotspots",
        "unit": "10^3 POIs",
        "outside_col": "function_exposure_uncovered",
        "total_col": "function_exposure_total",
        "value_scale": 1000.0,
        "axis_label": "10$^3$ POIs",
    },
]

COL_SPECS = [
    ("value", "baseline", "Top 10 by outside value\nBaseline"),
    ("value", "stress", "Top 10 by outside value\nTail-rainfall stress"),
    ("ratio", "baseline", "Top 10 by outside ratio\nBaseline"),
    ("ratio", "stress", "Top 10 by outside ratio\nTail-rainfall stress"),
]


def progress(msg: str) -> None:
    print(f"[fig5-coverage] {msg}", flush=True)


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def safe_div(numer: pd.Series | np.ndarray, denom: pd.Series | np.ndarray) -> np.ndarray:
    numer_arr = np.asarray(numer, dtype=float)
    denom_arr = np.asarray(denom, dtype=float)
    out = np.full_like(numer_arr, np.nan, dtype=float)
    mask = denom_arr > 0
    out[mask] = numer_arr[mask] / denom_arr[mask]
    return out


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def compact_city_label(value) -> str:
    if pd.isna(value):
        return ""
    label = str(value).strip()
    label = label[:5] if len(label) > 5 else label
    if lazy_pinyin is None:
        return label
    return "".join(lazy_pinyin(label)).title()

def load_gdp(gdp_csv: str | Path) -> pd.DataFrame:
    path = Path(gdp_csv)
    if not path.exists():
        progress(f"GDP file not found: {path}; using median color for all cities")
        return pd.DataFrame({"city_std": [], "gdp_per_capita": [], "region": []})
    gdp = read_csv(path)
    if "city_clean" not in gdp.columns:
        raise ValueError(f"GDP file must contain city_clean: {path}")
    gdp["city_std"] = gdp["city_clean"].map(cg.clean_city_name)
    if "gdp_per_capita" not in gdp.columns:
        if {"gdp_total", "resident_population"}.issubset(gdp.columns):
            gdp["gdp_per_capita"] = safe_numeric(gdp["gdp_total"]) / safe_numeric(gdp["resident_population"])
        else:
            gdp["gdp_per_capita"] = np.nan
    keep = ["city_std", "gdp_per_capita"]
    if "region" in gdp.columns:
        keep.append("region")
    return gdp[keep].drop_duplicates("city_std")


def load_primary_abs(
    event_abs_csv: str | Path,
    full_table_csv: str | Path,
    road_base_csv: str | Path,
) -> pd.DataFrame:
    path = Path(event_abs_csv)
    if path.exists():
        progress(f"loading Fig.4 event absolute exposure table: {path}")
        out = read_csv(path)
        out["Event_ID"] = out["Event_ID"].astype(int)
        return out
    progress("Fig.4 event table missing; rebuilding it before Fig.5")
    return fig4.load_or_build_primary_abs(path, full_table_csv, road_base_csv, recompute=False)


def add_tail_rainfall_weight(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = events.copy()
    rain = safe_numeric(out["Event_Peak_Rain"])
    q75 = float(rain.quantile(0.75))
    q95 = float(rain.quantile(0.95))
    denom = max(q95 - q75, 1e-9)
    tail = ((rain - q75) / denom).clip(lower=0, upper=1)
    out["tail_rainfall_weight"] = 1.0 + tail
    meta = {
        "rain_q75": q75,
        "rain_q95": q95,
        "stress_weight_min": float(out["tail_rainfall_weight"].min()),
        "stress_weight_median": float(out["tail_rainfall_weight"].median()),
        "stress_weight_max": float(out["tail_rainfall_weight"].max()),
    }
    return out, meta


def build_city_stress_ranking(
    event_abs: pd.DataFrame,
    gdp_csv: str | Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    events, stress_meta = add_tail_rainfall_weight(event_abs)
    gdp = load_gdp(gdp_csv)

    rows: list[pd.DataFrame] = []
    for spec in ROW_SPECS:
        usecols = [
            "city_std",
            "city_clean",
            "Event_ID",
            "total_reports",
            "total_affected_grids",
            "Event_Peak_Rain",
            "tail_rainfall_weight",
            spec["outside_col"],
            spec["total_col"],
        ]
        usecols = list(dict.fromkeys(usecols))
        work = events[usecols].copy()
        work["_outside"] = safe_numeric(work[spec["outside_col"]]).fillna(0).clip(lower=0)
        work["_total"] = safe_numeric(work[spec["total_col"]]).fillna(0).clip(lower=0)
        work["_w"] = safe_numeric(work["tail_rainfall_weight"]).fillna(1.0).clip(lower=1.0)
        work["_outside_w"] = work["_outside"] * work["_w"]
        work["_total_w"] = work["_total"] * work["_w"]

        city = (
            work.groupby(["city_std", "city_clean"], as_index=False)
            .agg(
                n_extreme_events=("Event_ID", "nunique"),
                total_reports=("total_reports", "sum"),
                total_affected_grids=("total_affected_grids", "sum"),
                baseline_value=("_outside", "sum"),
                baseline_total=("_total", "sum"),
                stress_value=("_outside_w", "sum"),
                stress_total=("_total_w", "sum"),
                mean_peak_rain=("Event_Peak_Rain", "mean"),
                max_peak_rain=("Event_Peak_Rain", "max"),
                mean_tail_weight=("_w", "mean"),
                max_tail_weight=("_w", "max"),
            )
        )
        city["metric"] = spec["metric"]
        city["metric_label"] = spec["label"].replace("\n", " ")
        city["unit"] = spec["unit"]
        city["value_scale"] = spec["value_scale"]
        city["baseline_ratio"] = safe_div(city["baseline_value"], city["baseline_total"]) * 100
        city["stress_ratio"] = safe_div(city["stress_value"], city["stress_total"]) * 100
        city["baseline_value_plot"] = city["baseline_value"] / spec["value_scale"]
        city["stress_value_plot"] = city["stress_value"] / spec["value_scale"]
        rows.append(city)

    ranking = pd.concat(rows, ignore_index=True)
    ranking = ranking.merge(gdp, on="city_std", how="left")
    median_gdp = float(safe_numeric(ranking["gdp_per_capita"]).median())
    ranking["gdp_missing"] = ranking["gdp_per_capita"].isna()
    ranking["gdp_per_capita"] = safe_numeric(ranking["gdp_per_capita"]).fillna(median_gdp)
    ranking["city_label"] = ranking["city_std"].map(compact_city_label)

    for metric, sub_idx in ranking.groupby("metric").groups.items():
        idx = list(sub_idx)
        ranking.loc[idx, "baseline_rank_value"] = (
            ranking.loc[idx].sort_values(["baseline_value", "baseline_ratio"], ascending=[False, False])
            .reset_index()
            .reset_index()
            .set_index("index")["level_0"]
            + 1
        )
        ranking.loc[idx, "stress_rank_value"] = (
            ranking.loc[idx].sort_values(["stress_value", "stress_ratio"], ascending=[False, False])
            .reset_index()
            .reset_index()
            .set_index("index")["level_0"]
            + 1
        )
        ranking.loc[idx, "baseline_rank_ratio"] = (
            ranking.loc[idx].sort_values(["baseline_ratio", "baseline_value"], ascending=[False, False])
            .reset_index()
            .reset_index()
            .set_index("index")["level_0"]
            + 1
        )
        ranking.loc[idx, "stress_rank_ratio"] = (
            ranking.loc[idx].sort_values(["stress_ratio", "stress_value"], ascending=[False, False])
            .reset_index()
            .reset_index()
            .set_index("index")["level_0"]
            + 1
        )

        base_value_top = set(ranking.loc[idx].nsmallest(10, "baseline_rank_value")["city_std"])
        stress_value_top = set(ranking.loc[idx].nsmallest(10, "stress_rank_value")["city_std"])
        base_ratio_top = set(ranking.loc[idx].nsmallest(10, "baseline_rank_ratio")["city_std"])
        stress_ratio_top = set(ranking.loc[idx].nsmallest(10, "stress_rank_ratio")["city_std"])
        ranking.loc[idx, "emerging_value_flag"] = ranking.loc[idx, "city_std"].isin(stress_value_top - base_value_top)
        ranking.loc[idx, "emerging_ratio_flag"] = ranking.loc[idx, "city_std"].isin(stress_ratio_top - base_ratio_top)

    return ranking, stress_meta


def get_top10(df: pd.DataFrame, mode: str, scenario: str) -> pd.DataFrame:
    if mode not in {"value", "ratio"}:
        raise ValueError(mode)
    if scenario not in {"baseline", "stress"}:
        raise ValueError(scenario)
    metric_col = f"{scenario}_{mode}"
    value_col = f"{scenario}_value"
    ratio_col = f"{scenario}_ratio"
    if mode == "value":
        sort_cols = [metric_col, ratio_col]
    else:
        sort_cols = [metric_col, value_col]
    sub = df.copy()
    for col in {metric_col, value_col, ratio_col}:
        sub[col] = safe_numeric(sub[col]).replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=[metric_col]).copy()
    sub = sub[sub[value_col] > 0].copy()
    return sub.sort_values(sort_cols, ascending=[False, False]).head(10).copy()


def style_panel(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y", linestyle="--", alpha=0.32, linewidth=0.65, color=COLORS["grid"])
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    ax.tick_params(width=0.8, length=3.0, color="#333333", pad=1.5)


def draw_panel(
    ax: plt.Axes,
    metric_df: pd.DataFrame,
    row_idx: int,
    col_idx: int,
    row_spec: dict,
    mode: str,
    scenario: str,
    col_title: str,
    norm: Normalize,
) -> pd.DataFrame:
    style_panel(ax)
    top = get_top10(metric_df, mode, scenario)
    emerging_col = "emerging_value_flag" if mode == "value" else "emerging_ratio_flag"
    rank_col = f"{scenario}_{mode}"
    value_col = f"{scenario}_value_plot"
    ratio_col = f"{scenario}_ratio"
    if mode == "value":
        top = top.sort_values([value_col, ratio_col], ascending=[True, True]).reset_index(drop=True)
    else:
        top = top.sort_values([ratio_col, value_col], ascending=[True, True]).reset_index(drop=True)

    x = np.arange(len(top))
    bar_vals = top[value_col].to_numpy(float)
    ratios = top[ratio_col].to_numpy(float)
    ymax = float(np.nanmax(bar_vals)) if len(bar_vals) else 1.0
    colors = [GDP_CMAP(norm(v)) for v in top["gdp_per_capita"].to_numpy(float)]
    ax.bar(x, bar_vals, color=colors, edgecolor=COLORS["bar_edge"], linewidth=0.55, width=0.72, zorder=2)

    show_ratio_dots = mode == "value"
    ax2 = ax.twinx()
    emerging = np.zeros(len(top), dtype=bool)
    if scenario == "stress":
        emerging = top[emerging_col].map(lambda v: bool(v) if pd.notna(v) else False).to_numpy(dtype=bool)

    if show_ratio_dots:
        ax2.scatter(x, ratios, color=COLORS["dot"], s=14, zorder=5)
        if scenario == "stress":
            ax2.scatter(
                x[emerging],
                ratios[emerging],
                color=COLORS["stress"],
                marker="^",
                s=34,
                zorder=6,
            )
    elif scenario == "stress" and emerging.any():
        marker_y = bar_vals[emerging] + max(ymax * 0.035, 0.5)
        ax.scatter(
            x[emerging],
            marker_y,
            color=COLORS["stress"],
            marker="^",
            s=34,
            zorder=6,
            clip_on=False,
        )

    labels = top["city_label"].astype(str).tolist()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=5.9)

    ax.set_ylim(0, max(ymax * 1.22, 1.0))
    ax2.set_ylim(0, 108)
    ax2.set_yticks([0, 50, 100])

    if row_idx == 0:
        ax.set_title(col_title, pad=8)
    if col_idx == 0:
        ax.set_ylabel(row_spec["axis_label"], fontsize=7.2)
    else:
        ax.set_yticklabels([])
    if show_ratio_dots and col_idx == 1:
        ax2.set_ylabel("Outside ratio (%)", fontsize=7.2)
    elif show_ratio_dots:
        ax2.set_yticklabels([])
    else:
        ax2.set_yticks([])
        ax2.set_yticklabels([])

    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    if show_ratio_dots:
        ax2.spines["right"].set_linewidth(0.8)
        ax2.spines["right"].set_color("#333333")
    else:
        ax2.spines["right"].set_visible(False)
    ax2.tick_params(width=0.8, length=2.7, color="#333333", pad=1.5)

    letter = chr(ord("a") + row_idx * 4 + col_idx)
    ax.text(
        0.015,
        1.02,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )

    top = top.copy()
    top["panel_letter"] = letter
    top["ranking_mode"] = mode
    top["scenario"] = scenario
    top["ranking_value"] = top[rank_col]
    top["plot_value"] = top[value_col]
    top["plot_ratio"] = top[ratio_col]
    return top


def plot_fig5(
    output_dir: str | Path = REPO_ROOT / "outputs" / "figures",
    output_stem: str = "Figure5_coverage_gap_priority_real",
    event_abs_csv: str | Path = FIG4_DIR / "Figure4_coverage_gap_exposure_real_event_abs.csv",
    full_table_csv: str | Path = cg.DEFAULT_FULL_TABLE,
    road_base_csv: str | Path = cg.DEFAULT_ROAD_BASE,
    gdp_csv: str | Path = REPO_ROOT / "data" / "gdp2020.csv",
    show: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_abs = load_primary_abs(event_abs_csv, full_table_csv, road_base_csv)
    ranking, stress_meta = build_city_stress_ranking(event_abs, gdp_csv)

    ranking_csv = output_dir / f"{output_stem}_city_stress_ranking.csv"
    top10_csv = output_dir / f"{output_stem}_top10_panels.csv"
    meta_txt = output_dir / f"{output_stem}_stress_definition.txt"
    ranking.to_csv(ranking_csv, index=False, encoding="utf-8-sig")

    gdp_vals = ranking["gdp_per_capita"].replace([np.inf, -np.inf], np.nan).dropna()
    norm = Normalize(vmin=float(gdp_vals.quantile(0.05)), vmax=float(gdp_vals.quantile(0.95)))

    fig = plt.figure(figsize=(14.8, 17.2))
    fig.patch.set_facecolor("white")

    L, R, T, B = 0.070, 0.965, 0.915, 0.115
    nrows, ncols = 4, 4
    wgap, hgap = 0.028, 0.040
    panel_w = (R - L - wgap * (ncols - 1)) / ncols
    panel_h = (T - B - hgap * (nrows - 1)) / nrows

    top_rows: list[pd.DataFrame] = []
    for r, row_spec in enumerate(ROW_SPECS):
        metric_df = ranking[ranking["metric"] == row_spec["metric"]].copy()
        for c, (mode, scenario, title) in enumerate(COL_SPECS):
            x0 = L + c * (panel_w + wgap)
            y0 = T - (r + 1) * panel_h - r * hgap
            ax = fig.add_axes([x0, y0, panel_w, panel_h])
            top = draw_panel(ax, metric_df, r, c, row_spec, mode, scenario, title, norm)
            top_rows.append(top)
            if c == 0:
                fig.text(
                    L - 0.045,
                    y0 + panel_h / 2,
                    row_spec["label"],
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    color=COLORS["text"],
                )

    panel_top10 = pd.concat(top_rows, ignore_index=True)
    panel_top10.to_csv(top10_csv, index=False, encoding="utf-8-sig")

    legend_ax = fig.add_axes([L, 0.944, R - L, 0.038])
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=5.8, label="Emerging coverage-gap priority"),
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=4.8, label="Outside ratio"),
    ]
    legend = legend_ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.00, 0.50),
        frameon=False,
        ncol=2,
        fontsize=8.4,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    legend_ax.add_artist(legend)
    legend_ax.text(0.31, 0.50, "GDP per capita", ha="left", va="center", fontsize=8.4)
    grad_ax = fig.add_axes([L + 0.365, 0.955, 0.120, 0.011])
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    grad_ax.imshow(grad, aspect="auto", cmap=GDP_CMAP, origin="lower")
    grad_ax.set_xticks([])
    grad_ax.set_yticks([])
    for spine in grad_ax.spines.values():
        spine.set_visible(False)
    fig.text(L + 0.358, 0.960, "Low", ha="right", va="center", fontsize=7.8, color=COLORS["gray"])
    fig.text(L + 0.490, 0.960, "High", ha="left", va="center", fontsize=7.8, color=COLORS["gray"])

    note = (
        "Baseline = observed extreme-event reported impact. Tail-rainfall stress reweights event exposure by peak-rainfall upper-tail intensity "
        f"(weight 1.0 below p75={stress_meta['rain_q75']:.1f}; linearly up to 2.0 at p95={stress_meta['rain_q95']:.1f}, capped above p95)."
    )
    fig.text(L, 0.055, note, ha="left", va="center", fontsize=7.5, color=COLORS["gray"])
    fig.text(
        L,
        0.035,
        "Bars show outside value; dots show outside ratio in value-ranked panels. Ratio-ranked panels contain cities with 100% outside ratio.",
        ha="left",
        va="center",
        fontsize=7.5,
        color=COLORS["gray"],
    )

    with open(meta_txt, "w", encoding="utf-8") as f:
        f.write(note + "\n")
        f.write("This is a stress-test / sensitivity scenario, not a climate projection.\n")
        for key, value in stress_meta.items():
            f.write(f"{key}: {value}\n")

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    progress(f"saved PNG: {png_path.resolve()}")
    progress(f"saved PDF: {pdf_path.resolve()}")
    progress(f"wrote ranking table: {ranking_csv.resolve()}")
    progress(f"wrote top10 panel table: {top10_csv.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_fig5(show=False)



