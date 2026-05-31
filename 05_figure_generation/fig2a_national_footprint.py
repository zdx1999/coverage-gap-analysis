#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import contextily as cx
except Exception:
    cx = None

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FULL_TABLE = REPO_ROOT / "data" / "city_event_grid_full_gaia_v4_poi_repaired_253cities.csv"
DEFAULT_SHP = REPO_ROOT / "data" / "shapefiles" / "china_prefecture.shp"
SIZE_FACTOR = 14.5
NATIONAL_OUTLINE_URL = "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"


def save_both(fig, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_pdf.with_suffix('.png')}")


def get_bubble_size(values):
    vals = pd.Series(values)
    return np.sqrt(np.clip(pd.to_numeric(vals, errors="coerce").fillna(0), 0, None)) * SIZE_FACTOR


def add_basemap_if_available(ax):
    if cx is None:
        return
    try:
        cx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=cx.providers.Esri.WorldTerrain,
            alpha=0.9,
            zorder=1,
            attribution=False,
        )
    except Exception:
        return


def add_north_arrow(ax, center_x=78.0, top_y=52.0):
    arrow_w = 1.2
    ax.add_patch(
        mpatches.Polygon(
            [[center_x, top_y], [center_x - arrow_w, top_y - 3], [center_x, top_y - 2.5]],
            facecolor="black",
            edgecolor="black",
            zorder=6,
        )
    )
    ax.add_patch(
        mpatches.Polygon(
            [[center_x, top_y], [center_x + arrow_w, top_y - 3], [center_x, top_y - 2.5]],
            facecolor="white",
            edgecolor="black",
            zorder=6,
        )
    )
    ax.text(center_x, top_y + 0.5, "N", ha="center", va="bottom", fontsize=16, fontweight="bold", zorder=6)


def add_scale_bar(ax, center_x=78.0, scale_y=46.0):
    scale_len = 9.0
    start_x = center_x - scale_len / 2
    end_x = center_x + scale_len / 2
    tick_h = 0.5
    ax.plot([start_x, end_x], [scale_y, scale_y], color="black", linewidth=1.5, zorder=6)
    ax.plot([start_x, start_x], [scale_y, scale_y + tick_h], color="black", linewidth=1.2, zorder=6)
    ax.plot([center_x, center_x], [scale_y, scale_y + tick_h * 0.6], color="black", linewidth=1.0, zorder=6)
    ax.plot([end_x, end_x], [scale_y, scale_y + tick_h], color="black", linewidth=1.2, zorder=6)
    ax.text(start_x, scale_y + 0.8, "0", ha="center", va="bottom", fontsize=9)
    ax.text(center_x, scale_y + 0.8, "500", ha="center", va="bottom", fontsize=9)
    ax.text(end_x, scale_y + 0.8, "1,000 km", ha="center", va="bottom", fontsize=9)


def load_national_outline():
    if gpd is None:
        return None
    try:
        gdf = gpd.read_file(NATIONAL_OUTLINE_URL)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326, allow_override=True)
        return gdf.to_crs(4326)
    except Exception:
        return None


def standardize_city_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().replace(" ", "")
    for suffix in ["市", "地区", "盟", "自治州", "特别行政区", "市辖区", "县"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def infer_city_name_col(gdf):
    preferred = ["地级", "地市", "NAME", "name", "CITY", "city", "市", "地名", "NAME_2", "市名称"]
    for c in gdf.columns:
        if c in preferred:
            return c
    for c in gdf.columns:
        cl = str(c).lower()
        if "name" in cl or "city" in cl:
            return c
    return None


def infer_peak_col(df):
    candidates = [
        "peak_rain",
        "peak_rainfall",
        "event_peak_rain",
        "peak_intensity",
        "rain_peak",
        "max_rain",
        "max_halfhour_rain",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {str(c).lower(): c for c in df.columns}
    for low, orig in lowered.items():
        if "peak" in low and "rain" in low:
            return orig
    return None


def infer_hotspot_cols(df):
    hotspot_col = None
    new_col = None
    for c in ["hotspot_refined", "hotspot_region", "is_hotspot", "hotspot"]:
        if c in df.columns:
            hotspot_col = c
            break
    for c in ["new_hotspot_region", "new_hotspot", "is_new_hotspot", "NewHotspot_ge"]:
        if c in df.columns:
            new_col = c
            break
    return hotspot_col, new_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-table", type=str, default=str(DEFAULT_FULL_TABLE))
    ap.add_argument("--shp", type=str, default=str(DEFAULT_SHP))
    ap.add_argument("--selected-city", type=str, default="郑州")
    ap.add_argument("--selected-event-id", type=int, default=5551)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if gpd is None:
        raise ImportError("geopandas is required for Figure 2a map.")

    df = pd.read_csv(args.full_table, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "city_clean" not in df.columns:
        raise ValueError("city_clean column not found.")

    df["city_std"] = df["city_clean"].map(standardize_city_name)
    for c in ["Event_ID", "flood_count", "is_extreme"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    peak_col = infer_peak_col(df)
    hotspot_col, new_col = infer_hotspot_cols(df)
    if peak_col is not None:
        df[peak_col] = pd.to_numeric(df[peak_col], errors="coerce")
    if hotspot_col is not None:
        df[hotspot_col] = pd.to_numeric(df[hotspot_col], errors="coerce")
    if new_col is not None:
        df[new_col] = pd.to_numeric(df[new_col], errors="coerce")

    event = df.groupby(["city_clean", "city_std", "Event_ID", "is_extreme"], as_index=False).agg(
        event_burden=("flood_count", "sum"),
        peak_rain=(peak_col, "max") if peak_col is not None else ("flood_count", "size"),
        hotspot_grids=(hotspot_col, "sum") if hotspot_col is not None else ("flood_count", "size"),
        new_hotspot_grids=(new_col, "sum") if new_col is not None else ("flood_count", "size"),
    )
    event["new_hotspot_share"] = np.where(
        event["hotspot_grids"] > 0,
        event["new_hotspot_grids"] / event["hotspot_grids"],
        np.nan,
    )

    city_all = event.groupby(["city_clean", "city_std"], as_index=False).agg(total_events=("Event_ID", "nunique")).copy()
    city_extreme = (
        event[event["is_extreme"] == 1]
        .groupby(["city_clean", "city_std"], as_index=False)
        .agg(
            extreme_events=("Event_ID", "nunique"),
            mean_new_hotspot_share=("new_hotspot_share", "mean"),
            max_peak_rain=("peak_rain", "max"),
        )
        .copy()
    )

    shp = gpd.read_file(args.shp)
    cname = infer_city_name_col(shp)
    if cname is None:
        raise ValueError("Could not infer city name column in shapefile.")
    shp["city_std"] = shp[cname].map(standardize_city_name)
    if shp.crs is None:
        shp = shp.set_crs(4326, allow_override=True)
    shp = shp.to_crs(4326)

    universe = shp[shp["city_std"].isin(city_all["city_std"])].copy()
    universe = universe.merge(city_all[["city_std", "total_events"]], on="city_std", how="left")
    city_extreme = city_extreme.merge(city_all[["city_std", "total_events"]], on="city_std", how="left")
    extreme = shp.merge(city_extreme, on="city_std", how="inner")
    selected = shp[shp["city_std"] == standardize_city_name(args.selected_city)].copy()
    national_outline = load_national_outline()

    universe["rep_x"] = universe.representative_point().x
    universe["rep_y"] = universe.representative_point().y
    extreme["rep_x"] = extreme.representative_point().x
    extreme["rep_y"] = extreme.representative_point().y
    if not selected.empty:
        selected["rep_x"] = selected.representative_point().x
        selected["rep_y"] = selected.representative_point().y

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    shp.plot(ax=ax, facecolor="#EAEAEA", edgecolor="none", zorder=1)
    if national_outline is not None:
        national_outline.plot(ax=ax, facecolor="none", edgecolor="#444444", linewidth=0.4, alpha=0.85, zorder=2)
    else:
        shp.boundary.plot(ax=ax, facecolor="none", edgecolor="#444444", linewidth=0.4, alpha=0.85, zorder=2)

    vmax_val = max(float(extreme["extreme_events"].quantile(0.95)) if not extreme.empty else 1.0, 1.0)
    norm = mcolors.Normalize(vmin=0, vmax=vmax_val)

    ax.scatter(
        universe["rep_x"],
        universe["rep_y"],
        s=get_bubble_size(universe["total_events"]) * 1.75,
        facecolors="#d9d9d9",
        edgecolors="white",
        linewidths=0.25,
        alpha=0.65,
        zorder=3,
    )

    scatter = ax.scatter(
        extreme["rep_x"],
        extreme["rep_y"],
        s=get_bubble_size(extreme["total_events"]) * 1.75,
        c=extreme["extreme_events"],
        cmap=plt.cm.Reds,
        norm=norm,
        alpha=0.8,
        edgecolors="#330000",
        linewidths=0.5,
        zorder=4,
    )

    if not selected.empty:
        city_display_name = "Zhengzhou" if standardize_city_name(args.selected_city) == "郑州" else args.selected_city
        ax.scatter(selected["rep_x"], selected["rep_y"], s=150, facecolors="none", edgecolors="#E63946", linewidths=1.5, zorder=6)
        ax.scatter(selected["rep_x"], selected["rep_y"], s=40, facecolors="#1D3557", edgecolors="white", linewidths=0.6, zorder=7)
        ax.annotate(
            city_display_name,
            xy=(selected["rep_x"].iloc[0], selected["rep_y"].iloc[0]),
            xytext=(selected["rep_x"].iloc[0] + 4.4, selected["rep_y"].iloc[0] + 2.0),
            fontsize=12,
            fontweight="bold",
            color="#111111",
            arrowprops=dict(arrowstyle="-", color="#333333", lw=0.8),
            zorder=8,
        )

    n_all = int(city_all["city_std"].nunique())
    n_ext = int(city_extreme["city_std"].nunique())
    n_evt = int(event[event["is_extreme"] == 1]["Event_ID"].nunique())
    txt = f"{n_all} cities in universe\n{n_ext} cities with extreme events\n{n_evt} extreme events"
    ax.text(
        75.6,
        31.2,
        txt,
        fontsize=9.0,
        ha="left",
        va="top",
        bbox=dict(boxstyle="square,pad=0.28", facecolor="white", edgecolor="#bbbbbb", linewidth=0.6, alpha=0.95),
        zorder=6,
    )

    legend_sizes = [5, 10, 20, 30]
    handles = []
    for v in legend_sizes:
        handles.append(ax.scatter([], [], s=float(get_bubble_size([v]).iloc[0] * 1.75), color="none", edgecolor="#333333", linewidth=0.8))
    leg = ax.legend(
        handles,
        [f"{v}" for v in legend_sizes],
        title="Events per City",
        title_fontsize=10,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0.08, 0.13),
        frameon=False,
        labelspacing=1.2,
        borderpad=0,
        scatterpoints=1,
    )
    leg.get_title().set_fontweight("bold")

    cax = ax.inset_axes([0.035, 0.12, 0.015, 0.20])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("Number of Extreme Events", fontsize=10, fontweight="bold", labelpad=8)
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    cbar.outline.set_linewidth(0.5)

    add_north_arrow(ax)
    add_scale_bar(ax)

    ax_inset = ax.inset_axes([0.80, 0.05, 0.12, 0.20])
    if national_outline is not None:
        national_outline.plot(ax=ax_inset, facecolor="#EAEAEA", edgecolor="none", zorder=1)
        national_outline.boundary.plot(ax=ax_inset, edgecolor="#333333", linewidth=0.85, zorder=2)
    else:
        shp.plot(ax=ax_inset, facecolor="#EAEAEA", edgecolor="none", zorder=1)
        shp.boundary.plot(ax=ax_inset, edgecolor="#333333", linewidth=0.85, zorder=2)

    ax_inset.scatter(
        universe["rep_x"],
        universe["rep_y"],
        s=get_bubble_size(universe["total_events"]) * 0.22,
        facecolors="#BBBBBB",
        edgecolors="none",
        alpha=0.5,
        zorder=3,
    )
    ax_inset.scatter(
        extreme["rep_x"],
        extreme["rep_y"],
        s=get_bubble_size(extreme["total_events"]) * 0.34,
        c=extreme["extreme_events"],
        cmap=plt.cm.YlOrRd,
        norm=norm,
        alpha=0.8,
        edgecolors="none",
        zorder=4,
    )
    ax_inset.set_xlim(106, 123)
    ax_inset.set_ylim(2, 25)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for side, spine in ax_inset.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor("#111111")
        spine.set_linewidth(1.2)

    ax.set_xlim(73, 136)
    ax.set_ylim(17, 54)
    ax.axis("off")
    ax.text(-0.02, 1.02, "a", transform=ax.transAxes, ha="left", va="bottom", fontsize=18, fontweight="bold")

    plt.tight_layout()
    save_both(fig, Path(args.out))


if __name__ == "__main__":
    main()
