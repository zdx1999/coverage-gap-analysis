#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge GEE extraction results back to GAIA-based city grid centroids.

Purpose
-------
This script is specifically for the GAIA workflow where your base grid file is:
    city_grid_base_centroids_gaia.csv

It merges directly on:
- grid_id (preferred), or
- fallback: city_clean + rounded centroid coordinates

Input
-----
1) base grid centroid csv:
   refined_outputs/grid_universe/city_grid_base_centroids_gaia.csv
2) GEE output csv from extract_ntl_pop_with_gee_api.py

Output
------
- refined_outputs/grid_universe/city_grid_base_centroids_gaia_with_gee.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

DEFAULT_REFINED = Path(r"D:/SM/zdxrun/datatest/research/outputs_spatial_burden/refined_outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-dir", type=str, default=str(DEFAULT_REFINED))
    parser.add_argument("--gee-csv", type=str, required=True, help="Path to GEE output CSV")
    parser.add_argument("--base-csv", type=str, default=None, help="Optional path to city_grid_base_centroids_gaia.csv")
    args = parser.parse_args()

    refined_dir = Path(args.refined_dir)
    grid_dir = refined_dir / "grid_universe"

    base_csv = Path(args.base_csv) if args.base_csv else (grid_dir / "city_grid_base_centroids_gaia.csv")
    gee_csv = Path(args.gee_csv)

    if not base_csv.exists():
        raise FileNotFoundError(f"Base grid CSV not found: {base_csv}")
    if not gee_csv.exists():
        raise FileNotFoundError(f"GEE CSV not found: {gee_csv}")

    base = pd.read_csv(base_csv)
    gee = pd.read_csv(gee_csv)

    print(f"Base rows: {len(base)}")
    print(f"GEE rows: {len(gee)}")

    # normalize potential output names from the API script
    rename_map = {}
    for src, dst in [
        ("longitude", "centroid_lon"),
        ("latitude", "centroid_lat"),
        ("ntl_avg_rad", "ntl_avg_rad_2021"),
        ("ntl_cf_cvg", "ntl_cf_cvg_2021"),
        ("worldpop", "worldpop_2020"),
    ]:
        if src in gee.columns and dst not in gee.columns:
            rename_map[src] = dst
    if rename_map:
        gee = gee.rename(columns=rename_map)

    # preferred merge: grid_id
    if "grid_id" in base.columns and "grid_id" in gee.columns:
        gee_sub = gee.drop_duplicates("grid_id").copy()
        merged = base.merge(gee_sub, on="grid_id", how="left", suffixes=("", "_gee"))
        merge_mode = "grid_id"
    else:
        required = {"city_clean", "centroid_lon", "centroid_lat"}
        if not required.issubset(set(base.columns)) or not required.issubset(set(gee.columns)):
            raise ValueError(
                "Need either grid_id in both tables, or (city_clean, centroid_lon, centroid_lat) in both tables."
            )

        base["centroid_lon_round"] = pd.to_numeric(base["centroid_lon"], errors="coerce").round(6)
        base["centroid_lat_round"] = pd.to_numeric(base["centroid_lat"], errors="coerce").round(6)
        gee["centroid_lon_round"] = pd.to_numeric(gee["centroid_lon"], errors="coerce").round(6)
        gee["centroid_lat_round"] = pd.to_numeric(gee["centroid_lat"], errors="coerce").round(6)

        gee_sub = gee.drop_duplicates(["city_clean", "centroid_lon_round", "centroid_lat_round"]).copy()
        merged = base.merge(
            gee_sub,
            on=["city_clean", "centroid_lon_round", "centroid_lat_round"],
            how="left",
            suffixes=("", "_gee")
        )
        merge_mode = "city+coords"

    for c in ["ntl_avg_rad_2021", "ntl_cf_cvg_2021", "worldpop_2020"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    out_csv = grid_dir / "city_grid_base_centroids_gaia_with_gee.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    matched_share = merged["ntl_avg_rad_2021"].notna().mean() if "ntl_avg_rad_2021" in merged.columns else 0.0
    print(f"Merge mode: {merge_mode}")
    print(f"Saved: {out_csv}")
    print(f"Matched share on NTL: {matched_share:.4f}")
    print(f"Output rows: {len(merged)}")


if __name__ == "__main__":
    main()
