#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Patch

try:
    import geopandas as gpd
except Exception:
    gpd = None

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

WEB_MERCATOR_R = 6378137.0
DISPLAY_INSET_M = 1.5
OVERLAP_INSET_M = 150.0

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FULL_TABLE = REPO_ROOT / "data" / "city_event_grid_full_gaia_v4_poi_repaired_253cities.csv"
DEFAULT_ROAD_BASE = REPO_ROOT / "data" / "city_grid_base_centroids_gaia_with_gee_terrain_repaired_253cities.csv"
DEFAULT_POI_BASE = REPO_ROOT / "data" / "grid_poi_baseline_2018_bytype.csv"
DEFAULT_SHP = REPO_ROOT / "data" / "shapefiles" / "china_prefecture.shp"


def save_both(fig, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def standardize_city_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().replace(" ", "")
    for suffix in ["市", "地区", "盟", "自治州", "特别行政区", "市辖区", "县"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
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


def load_boundary(city_name: str, shp_path: Path):
    if gpd is None or not shp_path.exists():
        return None
    try:
        gdf = gpd.read_file(shp_path)
    except Exception:
        return None
    ccol = infer_city_name_col(gdf)
    if ccol is None:
        return None
    gdf["city_std"] = gdf[ccol].map(standardize_city_name)
    target = standardize_city_name(city_name)
    hit = gdf[gdf["city_std"] == target].copy()
    if hit.empty:
        hit = gdf[gdf["city_std"].astype(str).str.contains(target, na=False)].copy()
    if hit.empty:
        return None
    if hit.crs is None:
        hit = hit.set_crs(4326, allow_override=True)
    try:
        hit = hit.to_crs(4326)
    except Exception:
        pass
    return hit


def pick_col(cols, exact_candidates, contains_groups):
    cols = list(cols)
    for c in exact_candidates:
        if c in cols:
            return c
    lowered = {str(c).lower(): c for c in cols}
    for group in contains_groups:
        for low, orig in lowered.items():
            if all(token in low for token in group):
                return orig
    return None


def get_hotspot_cols(df: pd.DataFrame):
    cols = list(df.columns)
    hotspot_col = pick_col(
        cols,
        ["hotspot_refined", "hotspot_region", "is_hotspot", "hotspot", "hotspot_binary"],
        [["hotspot"], ["is", "hotspot"]],
    )
    new_col = pick_col(
        cols,
        ["new_hotspot_region", "new_hotspot", "is_new_hotspot", "new_hotspot_binary", "NewHotspot_ge"],
        [["new", "hotspot"], ["new_hotspot"]],
    )
    return hotspot_col, new_col


def get_secondary_road_cols(df: pd.DataFrame):
    cols = list(df.columns)
    l2 = pick_col(
        cols,
        ["road_l2_len_km", "road_len_l2_km", "road_l2_km", "road_l2_length_km", "grip4_road_l2_len_km", "grip_l2_len_km", "l2_len_km", "road_len_km_grip4_l2"],
        [["road", "l2", "km"], ["road", "l2"], ["l2", "length"]],
    )
    l3 = pick_col(
        cols,
        ["road_l3_len_km", "road_len_l3_km", "road_l3_km", "road_l3_length_km", "grip4_road_l3_len_km", "grip_l3_len_km", "l3_len_km", "road_len_km_grip4_l3"],
        [["road", "l3", "km"], ["road", "l3"], ["l3", "length"]],
    )
    return l2, l3


def infer_poi_bucket_cols(df: pd.DataFrame):
    cols = list(df.columns)
    bucket_patterns = {
        "commercial_life": ["poi_commercial_life_2018", "poi_commercial_life", "commercial_life_poi", "commercial_life", "poi_life_service", "life_service", "poi_commercial"],
        "residential": ["poi_residential_2018", "poi_residential", "residential_poi", "residential"],
        "transport": ["poi_transport_2018", "poi_transport", "transport_poi", "transport"],
        "education_culture": ["poi_education_culture_2018", "poi_education_culture", "education_culture_poi", "education_culture", "poi_education", "education"],
    }
    out = {}
    lowered = {str(c).lower(): c for c in cols}
    for bucket, patterns in bucket_patterns.items():
        chosen = None
        for pat in patterns:
            for low, orig in lowered.items():
                if pat in low and not low.endswith("_absolute") and "share" not in low:
                    chosen = orig
                    break
            if chosen is not None:
                break
        if chosen is None:
            tokens = bucket.split("_")
            for low, orig in lowered.items():
                if all(tok in low for tok in tokens) and "share" not in low:
                    chosen = orig
                    break
        out[bucket] = chosen
    return out


def prepare_grid_keys(df: pd.DataFrame):
    out = df.copy()
    out["city_std"] = out["city_clean"].map(standardize_city_name)
    out["lon_r6"] = pd.to_numeric(out["centroid_lon"], errors="coerce").round(6)
    out["lat_r6"] = pd.to_numeric(out["centroid_lat"], errors="coerce").round(6)
    return out


def merge_base(event_df: pd.DataFrame, base_df: pd.DataFrame, value_cols):
    base_df = base_df.copy()
    base_df.columns = [str(c).strip() for c in base_df.columns]
    if "grid_id" in base_df.columns and "grid_id" in event_df.columns:
        base_df["grid_id"] = base_df["grid_id"].astype(str).str.strip()
        evt = event_df.copy()
        evt["grid_id"] = evt["grid_id"].astype(str).str.strip()
        keep_cols = ["grid_id"]
    else:
        if "city_clean" not in base_df.columns or "centroid_lon" not in base_df.columns or "centroid_lat" not in base_df.columns:
            raise ValueError("Need either grid_id in both files, or city_clean + centroid_lon + centroid_lat in base file.")
        base_df = prepare_grid_keys(base_df)
        evt = prepare_grid_keys(event_df)
        keep_cols = ["city_std", "lon_r6", "lat_r6"]
    base_keep = keep_cols + [c for c in value_cols if c is not None]
    base_keep = list(dict.fromkeys(base_keep))
    return evt.merge(base_df[base_keep].drop_duplicates(), on=keep_cols, how="left")


def add_projected_xy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lon = pd.to_numeric(out["centroid_lon"], errors="coerce").to_numpy(float)
    lat = pd.to_numeric(out["centroid_lat"], errors="coerce").to_numpy(float)
    lat = np.clip(lat, -85.05112878, 85.05112878)
    out["plot_x"] = WEB_MERCATOR_R * np.deg2rad(lon)
    out["plot_y"] = WEB_MERCATOR_R * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0))
    return out


def build_polygons(df: pd.DataFrame, inset_m: float = DISPLAY_INSET_M):
    polys = []
    for _, r in df.iterrows():
        cx = float(r["plot_x"])
        cy = float(r["plot_y"])
        half_m = max(float(r["cell_size_m"]) / 2.0 - inset_m, 1.0)
        polys.append(np.array([
            [cx - half_m, cy - half_m],
            [cx + half_m, cy - half_m],
            [cx + half_m, cy + half_m],
            [cx - half_m, cy + half_m],
        ]))
    return polys


def draw_grid_polygons(ax, polys, facecolor, edgecolor="none", alpha=1.0, lw=0.0, zorder=3):
    for poly in polys:
        ax.add_patch(Polygon(poly, closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=lw, zorder=zorder))


def bbox_from_df(df: pd.DataFrame, pad_ratio=0.12):
    xmin = df["plot_x"].min()
    xmax = df["plot_x"].max()
    ymin = df["plot_y"].min()
    ymax = df["plot_y"].max()
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    return {"xmin": xmin - dx * pad_ratio, "xmax": xmax + dx * pad_ratio, "ymin": ymin - dy * pad_ratio, "ymax": ymax + dy * pad_ratio}


def rect_overlap(a, b):
    dx = min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"])
    dy = min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"])
    if dx <= 0 or dy <= 0:
        return 0.0
    inter = dx * dy
    area_a = (a["xmax"] - a["xmin"]) * (a["ymax"] - a["ymin"])
    area_b = (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])
    return inter / min(area_a, area_b)


def shift_box(box, dx=0.0, dy=0.0):
    return {"xmin": box["xmin"] + dx, "xmax": box["xmax"] + dx, "ymin": box["ymin"] + dy, "ymax": box["ymax"] + dy}


def square_box(box, size=None):
    cx = (box["xmin"] + box["xmax"]) / 2.0
    cy = (box["ymin"] + box["ymax"]) / 2.0
    if size is None:
        size = max(box["xmax"] - box["xmin"], box["ymax"] - box["ymin"])
    half = size / 2.0
    return {"xmin": cx - half, "xmax": cx + half, "ymin": cy - half, "ymax": cy + half}


def box_mask(df: pd.DataFrame, box):
    return (
        (df["plot_x"] >= box["xmin"]) & (df["plot_x"] <= box["xmax"]) &
        (df["plot_y"] >= box["ymin"]) & (df["plot_y"] <= box["ymax"])
    )


def label_anchor(box, xoff_frac=0.0, yoff_frac=0.03):
    w = box["xmax"] - box["xmin"]
    h = box["ymax"] - box["ymin"]
    return box["xmin"] + w * xoff_frac, box["ymax"] + h * yoff_frac


def add_small_legend(ax):
    handles = [
        Patch(facecolor="#9a9a9a", edgecolor="#6b6b6b", linewidth=0.8, alpha=0.35, label="Dense support area"),
        Patch(facecolor="#8b1e1e", edgecolor="white", linewidth=0.5, label="Recurrent hotspot"),
        Patch(facecolor="#16b5de", edgecolor="white", linewidth=0.5, label="New hotspot"),
        Patch(facecolor="white", edgecolor="#f2c14e", linewidth=1.2, label="Overlap with support"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.06, 1.00),
        frameon=True,
        framealpha=0.96,
        borderpad=0.35,
        handlelength=1.0,
        fontsize=7.4,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    leg.get_frame().set_edgecolor("#bfbfbf")
    leg.get_frame().set_linewidth(0.6)


def draw_box_with_halo(ax, box, label):
    ax.add_patch(Rectangle((box["xmin"], box["ymin"]), box["xmax"] - box["xmin"], box["ymax"] - box["ymin"], fill=False, edgecolor="white", linewidth=2.6, linestyle="-", zorder=6))
    ax.add_patch(Rectangle((box["xmin"], box["ymin"]), box["xmax"] - box["xmin"], box["ymax"] - box["ymin"], fill=False, edgecolor="#5c5c5c", linewidth=1.2, linestyle=(0, (3, 2)), zorder=7))
    tx, ty = label_anchor(box, xoff_frac=0.02, yoff_frac=0.04)
    ax.text(tx, ty, label, fontsize=10.5, fontweight="bold", ha="left", va="bottom",
            bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.85), zorder=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-table", type=str, default=str(DEFAULT_FULL_TABLE))
    ap.add_argument("--road-base", type=str, default=str(DEFAULT_ROAD_BASE))
    ap.add_argument("--poi-base", type=str, default=str(DEFAULT_POI_BASE))
    ap.add_argument("--event-id", type=int, required=True)
    ap.add_argument("--city", type=str, default="郑州")
    ap.add_argument("--shp", type=str, default=str(DEFAULT_SHP))
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    full = pd.read_csv(args.full_table, low_memory=False)
    full.columns = [str(c).strip() for c in full.columns]
    for c in ["Event_ID", "centroid_lon", "centroid_lat", "cell_size_m", "flood_count"]:
        if c in full.columns:
            full[c] = pd.to_numeric(full[c], errors="coerce")
    if "grid_id" in full.columns:
        full["grid_id"] = full["grid_id"].astype(str).str.strip()
    if "city_clean" not in full.columns:
        raise ValueError("city_clean column not found in full table.")
    full["city_std"] = full["city_clean"].map(standardize_city_name)

    event_mask = pd.to_numeric(full["Event_ID"], errors="coerce") == int(args.event_id)
    city_std = standardize_city_name(args.city)
    event_df = full[event_mask & (full["city_std"] == city_std)].copy()
    if event_df.empty:
        event_df = full[event_mask].copy()
        if event_df.empty:
            raise ValueError(f"No rows found for Event {args.event_id}")
    city = str(event_df["city_clean"].mode().iloc[0])

    event_df = event_df[event_df["centroid_lon"].notna() & event_df["centroid_lat"].notna()].copy()
    event_df["cell_size_m"] = pd.to_numeric(event_df["cell_size_m"], errors="coerce").fillna(1000)

    hotspot_col, new_col = get_hotspot_cols(event_df)
    if hotspot_col is None or new_col is None:
        raise ValueError("Could not infer hotspot/new-hotspot columns from full table.")
    event_df["hotspot_flag"] = pd.to_numeric(event_df[hotspot_col], errors="coerce").fillna(0).astype(int)
    event_df["new_flag"] = pd.to_numeric(event_df[new_col], errors="coerce").fillna(0).astype(int)
    event_df["recurrent_flag"] = ((event_df["hotspot_flag"] == 1) & (event_df["new_flag"] != 1)).astype(int)
    event_df["affected_flag"] = (pd.to_numeric(event_df["flood_count"], errors="coerce").fillna(0) > 0).astype(int)

    road_base = pd.read_csv(args.road_base, low_memory=False)
    road_base.columns = [str(c).strip() for c in road_base.columns]
    l2_col, l3_col = get_secondary_road_cols(road_base)
    if l2_col is None and l3_col is None:
        raise ValueError("Could not infer L2/L3 road columns from road-base file.")
    road_df = merge_base(event_df, road_base, [l2_col, l3_col])
    sec = np.zeros(len(road_df), dtype=float)
    if l2_col is not None:
        sec += pd.to_numeric(road_df[l2_col], errors="coerce").fillna(0).clip(lower=0).values
    if l3_col is not None:
        sec += pd.to_numeric(road_df[l3_col], errors="coerce").fillna(0).clip(lower=0).values
    road_df["secondary_road"] = sec

    poi_base = pd.read_csv(args.poi_base, low_memory=False)
    poi_base.columns = [str(c).strip() for c in poi_base.columns]
    poi_cols = infer_poi_bucket_cols(poi_base)
    if all(v is None for v in poi_cols.values()):
        raise ValueError("Could not infer everyday-function POI columns from poi-base file.")
    poi_df = merge_base(event_df, poi_base, [v for v in poi_cols.values() if v is not None])
    everyday = np.zeros(len(poi_df), dtype=float)
    for c in poi_cols.values():
        if c is not None:
            everyday += pd.to_numeric(poi_df[c], errors="coerce").fillna(0).clip(lower=0).values
    poi_df["everyday_poi"] = everyday

    plot_df = poi_df.copy()
    plot_df["secondary_road"] = road_df["secondary_road"].values
    plot_df = add_projected_xy(plot_df)
    plot_df["poly_ix"] = np.arange(len(plot_df))
    all_polys = build_polygons(plot_df)
    overlap_polys = build_polygons(plot_df, inset_m=OVERLAP_INSET_M)

    rec = plot_df[plot_df["recurrent_flag"] == 1].copy()
    new = plot_df[plot_df["new_flag"] == 1].copy()
    affected_non = plot_df[(plot_df["affected_flag"] == 1) & (plot_df["hotspot_flag"] != 1)].copy()
    if rec.empty or new.empty:
        raise ValueError("Need both recurrent and new hotspots for this event.")

    east_new = new[(new["plot_x"] >= new["plot_x"].median()) & (new["plot_y"] >= new["plot_y"].median())].copy()
    if len(east_new) < 6:
        east_new = new.sort_values(["plot_x", "plot_y"], ascending=[False, False]).head(min(12, len(new))).copy()
    c_box0 = bbox_from_df(east_new, pad_ratio=0.10)

    transition = plot_df[
        (plot_df["plot_x"] >= rec["plot_x"].quantile(0.24)) &
        (plot_df["plot_x"] <= rec["plot_x"].quantile(0.78)) &
        (plot_df["plot_y"] >= rec["plot_y"].quantile(0.22)) &
        (plot_df["plot_y"] <= rec["plot_y"].quantile(0.78)) &
        ((plot_df["recurrent_flag"] == 1) | (plot_df["new_flag"] == 1))
    ].copy()
    if transition.empty:
        transition = pd.concat([rec, new]).drop_duplicates().copy()
    d_box0 = bbox_from_df(transition, pad_ratio=0.04)

    side = max(
        c_box0["xmax"] - c_box0["xmin"], c_box0["ymax"] - c_box0["ymin"],
        d_box0["xmax"] - d_box0["xmin"], d_box0["ymax"] - d_box0["ymin"],
    ) * 1.06
    c_box = square_box(c_box0, side)
    d_box = square_box(d_box0, side)
    if rect_overlap(c_box, d_box) > 0.16:
        c_box = shift_box(c_box, dx=0.28 * side)
        d_box = shift_box(d_box, dx=-0.10 * side)

    hotspot_union = pd.concat([rec, new]).drop_duplicates()
    main_box = bbox_from_df(hotspot_union, pad_ratio=0.36)

    boundary = load_boundary(city, Path(args.shp))
    if boundary is not None:
        try:
            boundary = boundary.to_crs(3857)
        except Exception:
            boundary = None

    fig = plt.figure(figsize=(15.4, 5.4), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.20, 0.92, 0.92], wspace=0.10)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[0, 2])

    mesh_edge = "#e3e3e3"
    affected_fill = "#f4f4f4"
    recurrent_fill = "#8b1e1e"
    new_fill = "#16b5de"
    road_fill = "#9c9c9c"
    poi_fill = "#9a9a9a"
    overlay_edge = "#f2c14e"
    boundary_edge = "#8c8c8c"
    boundary_fill = "#fafafa"

    if boundary is not None:
        boundary.plot(ax=ax_main, facecolor=boundary_fill, edgecolor=boundary_edge, linewidth=0.9, zorder=1)
    draw_grid_polygons(ax_main, all_polys, facecolor="none", edgecolor=mesh_edge, alpha=1.0, lw=0.10, zorder=2)
    if not affected_non.empty:
        draw_grid_polygons(ax_main, [all_polys[j] for j in affected_non["poly_ix"]], facecolor=affected_fill, edgecolor="white", alpha=0.70, lw=0.05, zorder=3)
    draw_grid_polygons(ax_main, [all_polys[j] for j in rec["poly_ix"]], facecolor=recurrent_fill, edgecolor="white", alpha=0.92, lw=0.10, zorder=4)
    draw_grid_polygons(ax_main, [all_polys[j] for j in new["poly_ix"]], facecolor=new_fill, edgecolor="white", alpha=0.95, lw=0.10, zorder=5)
    draw_box_with_halo(ax_main, c_box, "c")
    draw_box_with_halo(ax_main, d_box, "d")
    ax_main.set_xlim(main_box["xmin"], main_box["xmax"])
    ax_main.set_ylim(main_box["ymin"], main_box["ymax"])
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_aspect("equal", adjustable="box")
    for s in ax_main.spines.values():
        s.set_visible(False)
    ax_main.text(0.01, 0.99, "b", transform=ax_main.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")

    # c
    c_df = plot_df[box_mask(plot_df, c_box)].copy()
    c_df["road_dense"] = False
    road_pos = c_df["secondary_road"] > 0
    if road_pos.any():
        q2 = c_df.loc[road_pos, "secondary_road"].quantile(0.65)
        q3 = c_df.loc[road_pos, "secondary_road"].quantile(0.85)
        c_df.loc[c_df["secondary_road"] > q2, "road_dense"] = True
        c_df["road_q"] = 0
        c_df.loc[(c_df["secondary_road"] > q2) & (c_df["secondary_road"] <= q3), "road_q"] = 2
        c_df.loc[c_df["secondary_road"] > q3, "road_q"] = 3
    else:
        c_df["road_q"] = 0

    draw_grid_polygons(ax_c, [all_polys[j] for j in c_df["poly_ix"]], facecolor="none", edgecolor=mesh_edge, alpha=1.0, lw=0.10, zorder=1)
    c_dense_nonhot = c_df[(c_df["road_dense"]) & (c_df["hotspot_flag"] != 1)]
    c_dense_nonhot_q2 = c_dense_nonhot[c_dense_nonhot["road_q"] == 2]
    c_dense_nonhot_q3 = c_dense_nonhot[c_dense_nonhot["road_q"] == 3]
    if not c_dense_nonhot_q2.empty:
        draw_grid_polygons(ax_c, [all_polys[j] for j in c_dense_nonhot_q2["poly_ix"]], facecolor=road_fill, edgecolor="none", alpha=0.14, lw=0.0, zorder=2)
    if not c_dense_nonhot_q3.empty:
        draw_grid_polygons(ax_c, [all_polys[j] for j in c_dense_nonhot_q3["poly_ix"]], facecolor=road_fill, edgecolor="none", alpha=0.24, lw=0.0, zorder=3)
    c_rec_plain = c_df[(c_df["recurrent_flag"] == 1) & (~c_df["road_dense"])]
    c_rec_dense = c_df[(c_df["recurrent_flag"] == 1) & (c_df["road_dense"])]
    c_new_plain = c_df[(c_df["new_flag"] == 1) & (~c_df["road_dense"])]
    c_new_dense = c_df[(c_df["new_flag"] == 1) & (c_df["road_dense"])]
    if not c_rec_plain.empty:
        draw_grid_polygons(ax_c, [all_polys[j] for j in c_rec_plain["poly_ix"]], facecolor=recurrent_fill, edgecolor="white", alpha=0.92, lw=0.10, zorder=4)
    if not c_new_plain.empty:
        draw_grid_polygons(ax_c, [all_polys[j] for j in c_new_plain["poly_ix"]], facecolor=new_fill, edgecolor="white", alpha=0.95, lw=0.10, zorder=4)
    if not c_rec_dense.empty:
        draw_grid_polygons(ax_c, [all_polys[j] for j in c_rec_dense["poly_ix"]], facecolor=recurrent_fill, edgecolor="white", alpha=0.96, lw=0.10, zorder=5)
        draw_grid_polygons(ax_c, [overlap_polys[j] for j in c_rec_dense["poly_ix"]], facecolor="none", edgecolor=overlay_edge, alpha=1.0, lw=0.85, zorder=6)
    if not c_new_dense.empty:
        draw_grid_polygons(ax_c, [all_polys[j] for j in c_new_dense["poly_ix"]], facecolor=new_fill, edgecolor="white", alpha=0.98, lw=0.10, zorder=5)
        draw_grid_polygons(ax_c, [overlap_polys[j] for j in c_new_dense["poly_ix"]], facecolor="none", edgecolor=overlay_edge, alpha=1.0, lw=0.85, zorder=6)
    ax_c.set_xlim(c_box["xmin"], c_box["xmax"])
    ax_c.set_ylim(c_box["ymin"], c_box["ymax"])
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.set_aspect("equal", adjustable="box")
    for s in ax_c.spines.values():
        s.set_visible(False)
    ax_c.text(0.01, 0.99, "c", transform=ax_c.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")

    # d
    d_df = plot_df[box_mask(plot_df, d_box)].copy()
    d_df["poi_dense"] = False
    poi_pos = d_df["everyday_poi"] > 0
    if poi_pos.any():
        q2 = d_df.loc[poi_pos, "everyday_poi"].quantile(0.65)
        q3 = d_df.loc[poi_pos, "everyday_poi"].quantile(0.85)
        d_df.loc[d_df["everyday_poi"] > q2, "poi_dense"] = True
        d_df["poi_q"] = 0
        d_df.loc[(d_df["everyday_poi"] > q2) & (d_df["everyday_poi"] <= q3), "poi_q"] = 2
        d_df.loc[d_df["everyday_poi"] > q3, "poi_q"] = 3
    else:
        d_df["poi_q"] = 0

    draw_grid_polygons(ax_d, [all_polys[j] for j in d_df["poly_ix"]], facecolor="none", edgecolor=mesh_edge, alpha=1.0, lw=0.10, zorder=1)
    d_dense_nonhot = d_df[(d_df["poi_dense"]) & (d_df["hotspot_flag"] != 1)]
    d_dense_nonhot_q2 = d_dense_nonhot[d_dense_nonhot["poi_q"] == 2]
    d_dense_nonhot_q3 = d_dense_nonhot[d_dense_nonhot["poi_q"] == 3]
    if not d_dense_nonhot_q2.empty:
        draw_grid_polygons(ax_d, [all_polys[j] for j in d_dense_nonhot_q2["poly_ix"]], facecolor=poi_fill, edgecolor="none", alpha=0.14, lw=0.0, zorder=2)
    if not d_dense_nonhot_q3.empty:
        draw_grid_polygons(ax_d, [all_polys[j] for j in d_dense_nonhot_q3["poly_ix"]], facecolor=poi_fill, edgecolor="none", alpha=0.24, lw=0.0, zorder=3)
    d_rec_plain = d_df[(d_df["recurrent_flag"] == 1) & (~d_df["poi_dense"])]
    d_rec_dense = d_df[(d_df["recurrent_flag"] == 1) & (d_df["poi_dense"])]
    d_new_plain = d_df[(d_df["new_flag"] == 1) & (~d_df["poi_dense"])]
    d_new_dense = d_df[(d_df["new_flag"] == 1) & (d_df["poi_dense"])]
    if not d_rec_plain.empty:
        draw_grid_polygons(ax_d, [all_polys[j] for j in d_rec_plain["poly_ix"]], facecolor=recurrent_fill, edgecolor="white", alpha=0.92, lw=0.10, zorder=4)
    if not d_new_plain.empty:
        draw_grid_polygons(ax_d, [all_polys[j] for j in d_new_plain["poly_ix"]], facecolor=new_fill, edgecolor="white", alpha=0.95, lw=0.10, zorder=4)
    if not d_rec_dense.empty:
        draw_grid_polygons(ax_d, [all_polys[j] for j in d_rec_dense["poly_ix"]], facecolor=recurrent_fill, edgecolor="white", alpha=0.96, lw=0.10, zorder=5)
        draw_grid_polygons(ax_d, [overlap_polys[j] for j in d_rec_dense["poly_ix"]], facecolor="none", edgecolor=overlay_edge, alpha=1.0, lw=0.85, zorder=6)
    if not d_new_dense.empty:
        draw_grid_polygons(ax_d, [all_polys[j] for j in d_new_dense["poly_ix"]], facecolor=new_fill, edgecolor="white", alpha=0.98, lw=0.10, zorder=5)
        draw_grid_polygons(ax_d, [overlap_polys[j] for j in d_new_dense["poly_ix"]], facecolor="none", edgecolor=overlay_edge, alpha=1.0, lw=0.85, zorder=6)
    ax_d.set_xlim(d_box["xmin"], d_box["xmax"])
    ax_d.set_ylim(d_box["ymin"], d_box["ymax"])
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.set_aspect("equal", adjustable="box")
    for s in ax_d.spines.values():
        s.set_visible(False)
    ax_d.text(0.01, 0.99, "d", transform=ax_d.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
    add_small_legend(ax_d)

    print("Using road columns:")
    print(f"  L2: {l2_col}")
    print(f"  L3: {l3_col}")
    print("Using POI columns:")
    for k, v in poi_cols.items():
        print(f"  {k}: {v}")
    print("Using projected CRS: Web Mercator (EPSG:3857 display coordinates)")
    print("c and d boxes are forced to the same square size.")

    plt.tight_layout(rect=[0.0, 0.0, 0.92, 1.0])
    save_both(fig, Path(args.out))


if __name__ == "__main__":
    main()
