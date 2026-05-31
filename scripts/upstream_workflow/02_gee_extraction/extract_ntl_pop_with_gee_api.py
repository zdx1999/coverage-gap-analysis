#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract NTL and population for city-grid centroids using the Earth Engine Python API.

Input CSV must contain:
- grid_id
- city_clean
- centroid_lon
- centroid_lat

Default baseline:
- VIIRS monthly average radiance mean over 2021
- WorldPop 2020
"""

from __future__ import annotations
import argparse
from pathlib import Path
import math
import pandas as pd
import ee

DEFAULT_SCALE = 100
DEFAULT_CHUNK = 4000

def init_ee(project: str):
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)

def build_feature_chunk(df_chunk: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for _, row in df_chunk.iterrows():
        lon = float(row["centroid_lon"])
        lat = float(row["centroid_lat"])
        props = {
            "grid_id": str(row["grid_id"]),
            "city_clean": str(row["city_clean"]),
            "centroid_lon": lon,
            "centroid_lat": lat,
        }
        feats.append(ee.Feature(ee.Geometry.Point([lon, lat]), props))
    return ee.FeatureCollection(feats)

def build_stack(ntl_start: str, ntl_end: str, pop_year: int) -> ee.Image:
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(ntl_start, ntl_end)
    ntl_avg = viirs.select("avg_rad").mean().rename("ntl_avg_rad")
    ntl_cvg = viirs.select("cf_cvg").mean().rename("ntl_cf_cvg")

    worldpop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("year", pop_year))
        .mosaic()
        .rename("worldpop")
    )
    return ntl_avg.addBands(ntl_cvg).addBands(worldpop)

def sample_chunk(fc: ee.FeatureCollection, stack: ee.Image, scale: int) -> pd.DataFrame:
    sampled = stack.sampleRegions(
        collection=fc,
        scale=scale,
        geometries=False
    )
    df = ee.data.computeFeatures({
        "expression": sampled,
        "fileFormat": "PANDAS_DATAFRAME"
    })
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame(df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True, help="Google Cloud project for ee.Initialize(project=...)")
    parser.add_argument("--csv", type=str, required=True, help="Path to city_grid_base_centroids.csv")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--ntl-start", type=str, default="2021-01-01")
    parser.add_argument("--ntl-end", type=str, default="2022-01-01")
    parser.add_argument("--pop-year", type=int, default=2020)
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    args = parser.parse_args()

    in_csv = Path(args.csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    out_csv = Path(args.out) if args.out else in_csv.with_name("city_grid_base_centroids_ntl_pop_api.csv")

    df = pd.read_csv(in_csv)
    required = ["grid_id", "city_clean", "centroid_lon", "centroid_lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df["centroid_lon"] = pd.to_numeric(df["centroid_lon"], errors="coerce")
    df["centroid_lat"] = pd.to_numeric(df["centroid_lat"], errors="coerce")
    df = df.dropna(subset=["grid_id", "city_clean", "centroid_lon", "centroid_lat"]).copy()

    print(f"Rows to sample: {len(df)}")
    print(f"NTL window: {args.ntl_start} to {args.ntl_end}")
    print(f"Population year: {args.pop_year}")

    init_ee(args.project)
    stack = build_stack(args.ntl_start, args.ntl_end, args.pop_year)

    chunks = []
    n = len(df)
    chunk_size = int(args.chunk_size)
    n_chunks = math.ceil(n / chunk_size)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        sub = df.iloc[start:end].copy()
        print(f"Processing chunk {i+1}/{n_chunks}: rows {start} to {end-1}")
        fc = build_feature_chunk(sub)
        sampled_df = sample_chunk(fc, stack, args.scale)
        chunks.append(sampled_df)

    out = pd.concat(chunks, ignore_index=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv}")
    print(f"Output rows: {len(out)}")

if __name__ == "__main__":
    main()
