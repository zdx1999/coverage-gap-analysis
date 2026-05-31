#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-data Fig. 2 for the coverage-gap mainline.

Figure role:
    Cross-city evidence that historical flood hotspot inventories are
    informative but insufficient under extreme rainfall.

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
from matplotlib.patches import Patch

try:
    import contextily as ctx
except Exception:
    ctx = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
COVERAGE_DIR = REPO_ROOT / "scripts" / "coverage_gap_analysis"
SUPPORT_DIR = REPO_ROOT / "scripts" / "support"

for path in [COVERAGE_DIR, SUPPORT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fig2_zhengzhou_5551_final_refined_v5 as zz
import run_coverage_gap_analysis as cg


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


COLORS = {
    "panel_bg": "#FFFFFF",
    "city_fill": "#FAFAF8",
    "boundary": "#8A8F92",
    "grid_edge": "#D9D9D9",
    "builtup": "#F6F6F3",
    "context_mid": "#E4E8E8",
    "context_high": "#D3DBDD",
    "history": "#8B1E2D",
    "history_edge": "#5F1320",
    "covered": "#B65A5F",
    "uncovered": "#00A7A7",
    "support": "#C9911A",
    "support_fill": "#E7B83E",
    "report": "#1F1F1F",
    "text": "#222222",
    "gray_text": "#555555",
}

SELECTED_EVENTS = [
    ("Zhengzhou", "\u90d1\u5dde", 5551, "July 2021 extreme rainfall"),
    ("Guangzhou", "\u5e7f\u5dde", 4520, "Extreme rainfall event"),
    ("Beijing", "\u5317\u4eac", 7098, "Extreme rainfall event"),
    ("Wuhan", "\u6b66\u6c49", 3856, "Extreme rainfall event"),
    ("Fuzhou", "\u798f\u5dde", 7269, "Extreme rainfall event"),
    ("Tianjin", "\u5929\u6d25", 3309, "Extreme rainfall event"),
    ("Chengdu", "\u6210\u90fd", 3215, "Extreme rainfall event"),
]


def progress(msg: str) -> None:
    print(f"[fig2-coverage] {msg}", flush=True)


def style_panel(ax: plt.Axes, border: bool = True) -> None:
    ax.set_facecolor(COLORS["panel_bg"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(border)
        spine.set_color("#333333")
        spine.set_linewidth(0.95)


def style_stat_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.55, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.75)
    ax.tick_params(width=0.75, length=3.2, labelsize=8.4)


def hide_map_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_poly_list(ax: plt.Axes, polys, indices, facecolor, edgecolor, alpha, lw, zorder) -> None:
    selected = [polys[int(i)] for i in indices]
    if selected:
        zz.draw_grid_polygons(
            ax,
            selected,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            lw=lw,
            zorder=zorder,
        )


def draw_context_basemap(ax: plt.Axes, polys, visible: pd.DataFrame, large: bool) -> None:
    if visible.empty:
        return
    draw_poly_list(
        ax,
        polys,
        visible["poly_ix"],
        COLORS["builtup"],
        COLORS["grid_edge"],
        0.72,
        0.095 if not large else 0.12,
        1,
    )
    score = pd.to_numeric(visible.get("context_score", 0), errors="coerce").fillna(0)
    positive = visible[score > 0].copy()
    if positive.empty:
        return
    score_pos = pd.to_numeric(positive["context_score"], errors="coerce").fillna(0)
    q65 = score_pos.quantile(0.65)
    q85 = score_pos.quantile(0.85)
    mid = positive[(score_pos > q65) & (score_pos <= q85)]
    high = positive[score_pos > q85]
    draw_poly_list(ax, polys, mid["poly_ix"], COLORS["context_mid"], "none", 0.55, 0.0, 2)
    draw_poly_list(ax, polys, high["poly_ix"], COLORS["context_high"], "none", 0.68, 0.0, 3)


def draw_admin_boundary(
    ax: plt.Axes,
    boundary,
    large: bool,
    outline_only: bool = False,
    line_zorder: int = 4,
) -> None:
    if boundary is None:
        return
    if not outline_only:
        boundary.plot(
            ax=ax,
            facecolor=COLORS["city_fill"],
            edgecolor="none",
            linewidth=0,
            alpha=0.18 if large else 0.78,
            zorder=0,
        )
    try:
        boundary.boundary.plot(
            ax=ax,
            color=COLORS["boundary"],
            linewidth=0.82 if not large else 1.08,
            alpha=0.86,
            zorder=line_zorder,
        )
    except Exception:
        boundary.plot(
            ax=ax,
            facecolor="none",
            edgecolor=COLORS["boundary"],
            linewidth=0.82 if not large else 1.08,
            alpha=0.86,
            zorder=line_zorder,
        )


def add_online_basemap_legacy(ax: plt.Axes) -> None:
    print("姝ｅ湪璇锋眰鍦ㄧ嚎搴曞浘 (闇€瑕佽仈缃?...", flush=True)
    try:
        if ctx is None:
            raise ImportError("contextily is not available in the active Python environment")
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            zoom=11,
            alpha=0.9,
            attribution=False,
        )
    except Exception as e:
        print(f"搴曞浘鍔犺浇澶辫触: {e}", flush=True)
        ax.set_facecolor("#F0F0F0")


def add_online_basemap(ax: plt.Axes, alpha: float = 0.28, zoom: int = 11) -> bool:
    """Add a light Contextily/CartoDB basemap without built-in labels."""
    try:
        if ctx is None:
            raise ImportError("contextily is not available in the active Python environment")
        print("Requesting online basemap with Contextily CartoDB.PositronNoLabels...", flush=True)
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.PositronNoLabels,
            zoom=zoom,
            alpha=alpha,
            attribution=False,
        )
        return True
    except Exception as e:
        print(f"Panel-a basemap loading failed: {e}", flush=True)
    ax.set_facecolor("#F0F0F0")
    return False


def square_box_from_df(df: pd.DataFrame, pad_ratio: float = 0.18, min_span_m: float = 22000.0) -> dict:
    if df.empty:
        return {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    box = zz.bbox_from_df(df, pad_ratio=pad_ratio)
    side = max(box["xmax"] - box["xmin"], box["ymax"] - box["ymin"], min_span_m)
    return zz.square_box(box, side)


def expand_square_box(box: dict, factor: float = 1.25) -> dict:
    cx = (box["xmin"] + box["xmax"]) / 2
    cy = (box["ymin"] + box["ymax"]) / 2
    half = max(box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]) * factor / 2
    return {"xmin": cx - half, "xmax": cx + half, "ymin": cy - half, "ymax": cy + half}


def shift_box(box: dict, x_frac: float = 0.0, y_frac: float = 0.0) -> dict:
    width = box["xmax"] - box["xmin"]
    height = box["ymax"] - box["ymin"]
    dx = width * x_frac
    dy = height * y_frac
    return {
        "xmin": box["xmin"] + dx,
        "xmax": box["xmax"] + dx,
        "ymin": box["ymin"] + dy,
        "ymax": box["ymax"] + dy,
    }


def event_zoom_box(
    df: pd.DataFrame,
    pad_ratio: float = 0.14,
    min_span_m: float = 56000.0,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> dict:
    """Zoom to the main event-affected urban area while trimming far-flung outliers."""
    if df.empty:
        return square_box_from_df(df, pad_ratio=pad_ratio, min_span_m=min_span_m)
    x = pd.to_numeric(df["plot_x"], errors="coerce").dropna().to_numpy(float)
    y = pd.to_numeric(df["plot_y"], errors="coerce").dropna().to_numpy(float)
    if len(x) < 12 or len(y) < 12:
        return square_box_from_df(df, pad_ratio=pad_ratio, min_span_m=min_span_m)
    xmin, xmax = np.nanquantile(x, [q_low, q_high])
    ymin, ymax = np.nanquantile(y, [q_low, q_high])
    if not np.isfinite([xmin, xmax, ymin, ymax]).all() or xmin == xmax or ymin == ymax:
        return square_box_from_df(df, pad_ratio=pad_ratio, min_span_m=min_span_m)
    dx = xmax - xmin
    dy = ymax - ymin
    box = {
        "xmin": xmin - dx * pad_ratio,
        "xmax": xmax + dx * pad_ratio,
        "ymin": ymin - dy * pad_ratio,
        "ymax": ymax + dy * pad_ratio,
    }
    side = max(box["xmax"] - box["xmin"], box["ymax"] - box["ymin"], min_span_m)
    return zz.square_box(box, side)


def nice_scale_length(target_m: float) -> float:
    if not np.isfinite(target_m) or target_m <= 0:
        return 10000.0
    base = 10 ** np.floor(np.log10(target_m))
    for multiplier in [1, 2, 5, 10]:
        length = multiplier * base
        if length >= target_m:
            return float(length)
    return float(10 * base)


def draw_scale_bar(ax: plt.Axes, box: dict, large: bool) -> None:
    width = box["xmax"] - box["xmin"]
    height = box["ymax"] - box["ymin"]
    length = nice_scale_length(width * (0.14 if large else 0.18))
    x0 = box["xmin"] + width * 0.055
    y0 = box["ymin"] + height * 0.060
    tick = height * 0.010
    lw = 1.65 if large else 1.25
    ax.plot([x0, x0 + length], [y0, y0], color="#222222", linewidth=lw, solid_capstyle="butt", zorder=30)
    ax.plot([x0, x0], [y0 - tick, y0 + tick], color="#222222", linewidth=lw, zorder=30)
    ax.plot([x0 + length, x0 + length], [y0 - tick, y0 + tick], color="#222222", linewidth=lw, zorder=30)
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    ax.text(
        x0 + length / 2,
        y0 + height * 0.018,
        label,
        ha="center",
        va="bottom",
        fontsize=7.6 if large else 6.4,
        fontweight="bold",
        color="#222222",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
        zorder=31,
    )


def draw_north_arrow(ax: plt.Axes, large: bool) -> None:
    x = 0.952
    y = 0.925
    length = 0.060 if large else 0.070
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - length),
        xycoords=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.8 if large else 7.2,
        fontweight="bold",
        color="#222222",
        arrowprops=dict(
            arrowstyle="-|>",
            color="#222222",
            linewidth=1.2 if large else 1.0,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=10 if large else 8,
        ),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.35),
        zorder=32,
    )


def format_date_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if pd.isna(start) or pd.isna(end):
        return ""
    start = start.normalize()
    end = end.normalize()
    if start == end:
        return start.strftime("%Y-%m-%d")
    if start.year == end.year and start.month == end.month:
        return f"{start:%Y-%m-%d} to {end:%d}"
    if start.year == end.year:
        return f"{start:%Y-%m-%d} to {end:%m-%d}"
    return f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"


def format_event_date_label(sub: pd.DataFrame) -> str:
    times = []
    for col in ["first_flood_time", "last_flood_time"]:
        if col in sub.columns:
            times.append(pd.to_datetime(sub[col].astype(str), errors="coerce"))
    if not times:
        return ""
    dt = pd.concat(times).dropna()
    if dt.empty:
        return ""
    return format_date_range(dt.min(), dt.max())


def load_event_date_lookup(full_table: Path) -> dict[int, str]:
    header = pd.read_csv(full_table, nrows=0)
    required = ["Event_ID", "first_flood_time", "last_flood_time"]
    if any(c not in header.columns for c in required):
        return {}
    date_df = pd.read_csv(full_table, usecols=required, low_memory=False)
    date_df["Event_ID"] = pd.to_numeric(date_df["Event_ID"], errors="coerce")
    date_df = date_df.dropna(subset=["Event_ID"]).copy()
    date_df["Event_ID"] = date_df["Event_ID"].astype(int)
    date_df["first_dt"] = pd.to_datetime(date_df["first_flood_time"].astype(str), errors="coerce")
    date_df["last_dt"] = pd.to_datetime(date_df["last_flood_time"].astype(str), errors="coerce")
    grouped = date_df.groupby("Event_ID", observed=True).agg(
        first=("first_dt", "min"),
        last=("last_dt", "max"),
    )
    return {
        int(event_id): format_date_range(row["first"], row["last"])
        for event_id, row in grouped.iterrows()
    }


def load_full_with_history(full_table: Path, road_base: Path) -> pd.DataFrame:
    full = cg.load_full_table(full_table)
    road_lookup, _ = cg.load_road_lookup(road_base)
    full, _ = cg.attach_road(full, road_lookup)
    full["analysis_grid_id"] = full["grid_id"]
    progress("computing city-grid historical report totals")
    grid_total = (
        full.groupby(["city_std", "analysis_grid_id"], observed=True)["flood_count"]
        .sum()
        .rename("city_grid_total_reports")
        .reset_index()
    )
    return full.merge(grid_total, on=["city_std", "analysis_grid_id"], how="left")


def add_event_inventory_flags(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.copy()
    out["hist_reports_loo"] = (
        pd.to_numeric(out["city_grid_total_reports"], errors="coerce").fillna(0)
        - pd.to_numeric(out["flood_count"], errors="coerce").fillna(0)
    ).clip(lower=0)
    out["historical_inventory"] = False
    positive = out["hist_reports_loo"] > 0
    if positive.any():
        pos = out.loc[positive, ["hist_reports_loo", "analysis_grid_id"]].copy()
        pos = pos.sort_values(["hist_reports_loo", "analysis_grid_id"], ascending=[False, True])
        n_keep = max(1, int(np.ceil(len(pos) * 0.20)))
        out.loc[pos.head(n_keep).index, "historical_inventory"] = True
    out["affected_flag"] = pd.to_numeric(out["flood_count"], errors="coerce").fillna(0) > 0
    out["covered_impact"] = out["affected_flag"] & out["historical_inventory"]
    out["uncovered_impact"] = out["affected_flag"] & (~out["historical_inventory"])

    road = pd.to_numeric(out["secondary_road"], errors="coerce").fillna(0)
    poi = pd.to_numeric(out["poi_count_2018"], errors="coerce").fillna(0)
    pop = pd.to_numeric(out.get("worldpop_2020", 0), errors="coerce").fillna(0)
    out["context_score"] = np.log1p(pop.clip(lower=0)) + 0.85 * np.log1p(poi.clip(lower=0)) + 0.70 * np.log1p(road.clip(lower=0))
    road_dense = road > road[road > 0].quantile(0.70) if (road > 0).any() else pd.Series(False, index=out.index)
    poi_dense = poi > poi[poi > 0].quantile(0.70) if (poi > 0).any() else pd.Series(False, index=out.index)
    out["support_dense"] = road_dense | poi_dense
    return out


def prepare_event_data(
    full: pd.DataFrame,
    event_summary: pd.DataFrame,
    city_display: str,
    city_std: str,
    event_id: int,
    event_label: str,
    shp_path: str | Path,
    event_date_lookup: dict[int, str] | None = None,
) -> dict:
    sub = full[(full["city_std"] == city_std) & (full["Event_ID"].astype(int) == int(event_id))].copy()
    if sub.empty:
        raise ValueError(f"No rows found for {city_std} Event {event_id}")
    event_date_label = ""
    if event_date_lookup:
        event_date_label = event_date_lookup.get(int(event_id), "")
    if not event_date_label:
        event_date_label = format_event_date_label(sub)
    sub = add_event_inventory_flags(zz.add_projected_xy(sub))
    sub["poly_ix"] = np.arange(len(sub))
    all_polys = zz.build_polygons(sub)
    overlap_polys = zz.build_polygons(sub, inset_m=zz.OVERLAP_INSET_M)

    historical = sub[sub["historical_inventory"]].copy()
    covered = sub[sub["covered_impact"]].copy()
    uncovered = sub[sub["uncovered_impact"]].copy()
    affected = sub[sub["affected_flag"]].copy()

    if affected.empty:
        focus = sub.copy()
        view_box = square_box_from_df(focus, pad_ratio=0.10, min_span_m=42000.0)
    else:
        view_box = event_zoom_box(affected, pad_ratio=0.14, min_span_m=56000.0)
        if city_std == "北京" and int(event_id) == 7098:
            view_box = event_zoom_box(
                affected,
                pad_ratio=0.10,
                min_span_m=70000.0,
                q_low=0.10,
                q_high=0.90,
            )
        if city_std == "天津" and int(event_id) == 3309:
            view_box = event_zoom_box(
                affected,
                pad_ratio=0.10,
                min_span_m=44000.0,
                q_low=0.12,
                q_high=0.88,
            )
        if city_std == "福州" and int(event_id) == 7269:
            view_box = event_zoom_box(
                affected,
                pad_ratio=0.10,
                min_span_m=46000.0,
                q_low=0.10,
                q_high=0.90,
            )
            view_box = shift_box(view_box, x_frac=-0.12, y_frac=0.10)

    boundary = zz.load_boundary(str(sub["city_clean"].mode().iloc[0]), Path(shp_path))
    if boundary is not None:
        try:
            boundary = boundary.to_crs(3857)
        except Exception:
            boundary = None

    row = event_summary[(event_summary["city_std"] == city_std) & (event_summary["Event_ID"].astype(int) == int(event_id))]
    if row.empty:
        stats = {}
    else:
        stats = row.iloc[0].to_dict()

    return {
        "city_display": city_display,
        "city_std": city_std,
        "event_id": event_id,
        "event_label": event_label,
        "event_date_label": event_date_label,
        "plot_df": sub,
        "all_polys": all_polys,
        "overlap_polys": overlap_polys,
        "historical": historical,
        "covered": covered,
        "uncovered": uncovered,
        "affected": affected,
        "view_box": view_box,
        "boundary": boundary,
        "stats": stats,
    }


def draw_city_event_map(ax: plt.Axes, data: dict, letter: str, large: bool = False) -> None:
    style_panel(ax, border=True)
    df = data["plot_df"]
    polys = data["all_polys"]
    overlap_polys = data["overlap_polys"]
    box = data["view_box"]
    st = data["stats"]
    gap = 100 * float(st.get("uncovered_impact_share", np.nan))
    func = 100 * float(st.get("function_exposure_outside_historical_hotspots", np.nan))
    reports_n = int(float(st.get("total_reports", 0))) if st else 0
    date_label = data.get("event_date_label") or data.get("event_label", "")
    map_title = f"{data['city_display']}\n{date_label}\nReports {reports_n:,}"
    visible = df[zz.box_mask(df, box)].copy()
    if visible.empty:
        visible = df.copy()

    ax.set_xlim(box["xmin"], box["xmax"])
    ax.set_ylim(box["ymin"], box["ymax"])
    basemap_ok = add_online_basemap(ax, alpha=1.0, zoom=11 if large else 10)
    if not basemap_ok:
        ax.set_facecolor("#FFFFFF" if not large else "#F0F0F0")
    draw_admin_boundary(ax, data["boundary"], large, outline_only=basemap_ok, line_zorder=4)

    draw_context_basemap(ax, polys, visible, large)
    draw_poly_list(ax, polys, visible.loc[visible["historical_inventory"], "poly_ix"], COLORS["history"], COLORS["history_edge"], 0.64, 0.10, 5)
    draw_poly_list(ax, polys, visible.loc[visible["covered_impact"], "poly_ix"], COLORS["covered"], "white", 0.92, 0.08, 7)
    draw_poly_list(ax, polys, visible.loc[visible["uncovered_impact"], "poly_ix"], COLORS["uncovered"], "white", 0.90, 0.08, 8)

    support_impact = visible[(visible["support_dense"]) & (visible["affected_flag"])]
    if not support_impact.empty:
        zz.draw_grid_polygons(
            ax,
            [overlap_polys[int(i)] for i in support_impact["poly_ix"]],
            facecolor="none",
            edgecolor=COLORS["support"],
            alpha=0.62,
            lw=0.26 if not large else 0.38,
            zorder=9,
        )

    draw_admin_boundary(ax, data["boundary"], large, outline_only=True, line_zorder=18)

    reports = visible[visible["affected_flag"]].copy()
    if not reports.empty:
        sizes = 1.2 + np.sqrt(pd.to_numeric(reports["flood_count"], errors="coerce").fillna(1).clip(lower=1)) * (2.0 if large else 1.25)
        ax.scatter(
            reports["plot_x"],
            reports["plot_y"],
            s=sizes,
            color=COLORS["report"],
            alpha=0.22 if large else 0.18,
            edgecolors="none",
            zorder=10,
        )

    ax.text(
        0.025,
        0.975,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.5 if large else 10.8,
        fontweight="bold",
        color=COLORS["text"],
        zorder=20,
    )
    ax.text(
        0.025,
        0.918,
        map_title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8 if large else 6.7,
        fontweight="bold",
        color=COLORS["gray_text"],
        linespacing=1.12,
        bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none", alpha=0.76),
        zorder=20,
    )
    badge = f"Gap {gap:.0f}%\nFunction outside {func:.0f}%"
    ax.text(
        0.985,
        0.025,
        badge,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3 if large else 5.9,
        fontweight="bold",
        color=COLORS["text"],
        linespacing=1.02,
        bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="#BDBDBD", linewidth=0.45, alpha=0.92),
        zorder=21,
    )

    ax.set_xlim(box["xmin"], box["xmax"])
    ax.set_ylim(box["ymin"], box["ymax"])
    draw_scale_bar(ax, box, large)
    draw_north_arrow(ax, large)
    hide_map_axes(ax)


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.0,
        1.035,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.4,
        fontweight="bold",
        color=COLORS["text"],
    )


def draw_summary_panel(ax: plt.Axes, event_summary: pd.DataFrame, random_baseline: pd.DataFrame) -> None:
    style_panel(ax, border=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rng = np.random.default_rng(416)
    dist_cols = [
        ("uncovered_impact_share", "Uncovered\nimpact", COLORS["uncovered"]),
        ("population_exposure_outside_historical_hotspots", "Population\noutside", "#6FB1A3"),
        ("road_exposure_outside_historical_hotspots", "Road\noutside", "#7A8CC7"),
        ("function_exposure_outside_historical_hotspots", "Function\noutside", COLORS["support"]),
    ]

    ax_h1 = ax.inset_axes([0.035, 0.12, 0.295, 0.80])
    style_stat_axis(ax_h1)
    panel_letter(ax_h1, "h")
    vals = [event_summary[c].dropna().clip(0, 1).to_numpy() * 100 for c, _, _ in dist_cols]
    for i, values in enumerate(vals, start=1):
        q25, med, q75 = np.nanpercentile(values, [25, 50, 75])
        ax_h1.vlines(i, q25, q75, color="#111111", linewidth=1.05, alpha=0.80, zorder=3)
        ax_h1.hlines(med, i - 0.17, i + 0.17, color="#111111", linewidth=1.35, zorder=4)
        ax_h1.text(
            i + 0.20,
            min(med + 2.8, 104),
            f"{med:.0f}%",
            ha="left",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="#111111",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.35),
            zorder=5,
        )
        xj = rng.normal(i, 0.055, len(values))
        ax_h1.scatter(xj, values, s=4.6, color="#8E8E8E", alpha=0.22, edgecolors="none", zorder=2)
    ax_h1.set_ylim(-2, 108)
    ax_h1.set_ylabel("Event share (%)", fontsize=9.0)
    ax_h1.set_xticks(range(1, len(dist_cols) + 1), [label for _, label, _ in dist_cols], fontsize=8.3)

    ax_h2 = ax.inset_axes([0.382, 0.12, 0.275, 0.80])
    style_stat_axis(ax_h2)
    panel_letter(ax_h2, "i")
    rand = random_baseline.copy()
    rand["observed_pct"] = pd.to_numeric(rand["observed_impact_coverage_rate"], errors="coerce") * 100
    rand["random_pct"] = pd.to_numeric(rand["random_mean_impact_coverage_rate"], errors="coerce") * 100
    rand["sig"] = pd.to_numeric(rand["empirical_p_value"], errors="coerce") < 0.05
    obs_med = float(rand["observed_pct"].median())
    random_med = float(rand["random_pct"].median())
    lift_med = float(pd.to_numeric(rand["coverage_lift"], errors="coerce").replace([np.inf, -np.inf], np.nan).median())
    sig_share = float(rand["sig"].mean())
    x_max = max(10.0, float(rand["random_pct"].quantile(0.995)) * 1.12)
    x_max = min(15.0, x_max)
    ax_h2.scatter(
        rand.loc[~rand["sig"], "random_pct"],
        rand.loc[~rand["sig"], "observed_pct"],
        s=11,
        color="#BDBDBD",
        edgecolors="none",
        alpha=0.45,
        label="n.s.",
    )
    ax_h2.scatter(
        rand.loc[rand["sig"], "random_pct"],
        rand.loc[rand["sig"], "observed_pct"],
        s=13,
        color=COLORS["history"],
        edgecolors="white",
        linewidths=0.15,
        alpha=0.70,
        label="p < 0.05",
    )
    ax_h2.plot([0, x_max], [0, x_max], color="#222222", linewidth=0.9, linestyle="--")
    ax_h2.text(
        0.96,
        0.93,
        f"Historical median = {obs_med:.1f}%\nRandom median = {random_med:.1f}%\nMedian lift = {lift_med:.2f}x\n{sig_share:.1%} events p < 0.05",
        transform=ax_h2.transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#BDBDBD", boxstyle="square,pad=0.18", linewidth=0.45, alpha=0.92),
    )
    ax_h2.set_xlim(-0.25, x_max)
    ax_h2.set_ylim(-2, 110)
    ax_h2.set_xticks(np.arange(0, x_max + 0.01, 2))
    ax_h2.set_xlabel("Random baseline coverage (%)", fontsize=9.0)
    ax_h2.set_ylabel("Historical-inventory coverage (%)", fontsize=9.0)
    ax_h2.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.12),
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#BDBDBD",
        fontsize=6.9,
        handletextpad=0.25,
        borderpad=0.25,
        labelspacing=0.25,
    )

    ax_h3 = ax.inset_axes([0.705, 0.12, 0.260, 0.80])
    style_stat_axis(ax_h3)
    panel_letter(ax_h3, "j")
    x = event_summary["uncovered_impact_share"].clip(0, 1).to_numpy(float) * 100
    yv = event_summary["function_exposure_outside_historical_hotspots"].clip(0, 1).to_numpy(float) * 100
    valid = np.isfinite(x) & np.isfinite(yv)
    sizes = 5 + np.sqrt(event_summary.loc[valid, "total_affected_grids"].to_numpy(float).clip(1)) * 1.2
    ax_h3.scatter(x[valid], yv[valid], s=sizes, alpha=0.32, color=COLORS["uncovered"], edgecolors="none")
    if valid.sum() >= 3:
        coef = np.polyfit(x[valid], yv[valid], 1)
        xx = np.linspace(20, 100, 100)
        ax_h3.plot(xx, coef[0] * xx + coef[1], color="#111111", linewidth=1.2)
        pearson = pd.Series(x[valid]).corr(pd.Series(yv[valid]), method="pearson")
        spearman = pd.Series(x[valid]).corr(pd.Series(yv[valid]), method="spearman")
    else:
        pearson = np.nan
        spearman = np.nan
    ax_h3.text(
        0.05,
        0.90,
        f"Pearson r={pearson:.2f}\nSpearman rho={spearman:.2f}\nn={int(valid.sum())}",
        transform=ax_h3.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#BBBBBB", boxstyle="square,pad=0.18", linewidth=0.45),
    )
    ax_h3.set_xlim(20, 100)
    ax_h3.set_ylim(0, 105)
    ax_h3.set_xlabel("Uncovered impact share (%)", fontsize=9.0)
    ax_h3.set_ylabel("Function exposure outside (%)", fontsize=9.0)


def plot_fig2_coverage_gap(
    full_table: str | Path = cg.DEFAULT_FULL_TABLE,
    road_base: str | Path = cg.DEFAULT_ROAD_BASE,
    event_summary_csv: str | Path = COVERAGE_DIR / "outputs" / "event_level_coverage_gap.csv",
    random_baseline_csv: str | Path = COVERAGE_DIR / "outputs_validity" / "random_inventory_baseline.csv",
    shp_path: str | Path = zz.DEFAULT_SHP,
    output_dir: str | Path = REPO_ROOT / "outputs" / "figures",
    output_stem: str = "Figure2_coverage_gap_real",
    show: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_summary = pd.read_csv(event_summary_csv, encoding="utf-8-sig")
    event_summary = event_summary[event_summary["is_extreme"] == 1].copy()
    random_baseline = pd.read_csv(random_baseline_csv, encoding="utf-8-sig")
    event_date_lookup = load_event_date_lookup(Path(full_table))
    full = load_full_with_history(Path(full_table), Path(road_base))

    progress("preparing representative event maps")
    prepared = []
    for city_display, city_std, event_id, event_label in SELECTED_EVENTS:
        prepared.append(
            prepare_event_data(
                full,
                event_summary,
                city_display,
                city_std,
                event_id,
                event_label,
                shp_path,
                event_date_lookup=event_date_lookup,
            )
        )

    selected_rows = []
    for data in prepared:
        st = data["stats"].copy()
        st["panel_city"] = data["city_display"]
        selected_rows.append(st)
    pd.DataFrame(selected_rows).to_csv(output_dir / f"{output_stem}_selected_events.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(15.4, 10.25))
    fig.patch.set_facecolor("white")

    L, R, T, B = 0.045, 0.975, 0.960, 0.055
    summary_x, summary_y = L, B
    summary_w, summary_h = R - L, 0.320
    summary_gap = 0.012
    main_y = summary_y + summary_h + summary_gap
    main_h = T - main_y
    gap_x = 0.015
    big_x, big_y, big_w, big_h = L, main_y, 0.360, main_h
    right_x = big_x + big_w + gap_x
    right_w = R - right_x
    small_gap_x, small_gap_y = 0.012, 0.020
    small_w = (right_w - 2 * small_gap_x) / 3
    small_h = (main_h - small_gap_y) / 2

    ax_big = fig.add_axes([big_x, big_y, big_w, big_h])
    draw_city_event_map(ax_big, prepared[0], "a", large=True)

    for i, data in enumerate(prepared[1:]):
        row = i // 3
        col = i % 3
        x0 = right_x + col * (small_w + small_gap_x)
        y0 = main_y + main_h - (row + 1) * small_h - row * small_gap_y
        ax = fig.add_axes([x0, y0, small_w, small_h])
        draw_city_event_map(ax, data, chr(ord("b") + i), large=False)

    ax_sum = fig.add_axes([summary_x, summary_y, summary_w, summary_h])
    draw_summary_panel(ax_sum, event_summary, random_baseline)

    legend_handles = [
        Patch(facecolor=COLORS["history"], edgecolor=COLORS["history_edge"], label="Historical hotspot inventory"),
        Patch(facecolor=COLORS["covered"], edgecolor="white", label="Covered event impact"),
        Patch(facecolor=COLORS["uncovered"], edgecolor="white", label="Uncovered event impact"),
        Patch(facecolor="none", edgecolor=COLORS["support"], label="Function / support dense"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["report"], markeredgecolor="none", alpha=0.45, markersize=6.6, label="Flood reports"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=5,
        frameon=False,
        fontsize=8.2,
        handletextpad=0.34,
        columnspacing=0.70,
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
    plot_fig2_coverage_gap(show=False)




