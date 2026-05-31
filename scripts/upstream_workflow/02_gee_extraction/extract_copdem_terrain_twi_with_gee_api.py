#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract elevation, slope, and TWI proxy to GAIA-grid centroids using Earth Engine Python API.

Datasets
--------
1) Copernicus DEM GLO-30
   ee.ImageCollection("COPERNICUS/DEM/GLO30")
2) MERIT Hydro
   ee.Image("MERIT/Hydro/v1_0_1")

Variables
---------
- elevation_copdem_m
- slope_copdem_deg
- twi_proxy

TWI proxy formula
-----------------
twi_proxy = ln( upa_m2 / tan(slope_rad) )

where:
- upa is MERIT Hydro upstream drainage area in km^2, converted to m^2
- slope_rad is Copernicus slope in radians

This is a practical proxy for first-round controls, not a strict same-source Copernicus-only TWI.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import math
import pandas as pd
import ee

DEFAULT_SCALE = 90
DEFAULT_CHUNK = 3000


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


def build_stack() -> ee.Image:
    # Copernicus DEM: official note says slope should be computed after reprojection
    dem = (
        ee.ImageCollection("COPERNICUS/DEM/GLO30")
        .select("DEM")
        .mosaic()
        .setDefaultProjection("EPSG:3857", None, 30)
        .rename("elevation_copdem_m")
    )

    slope_deg = ee.Terrain.slope(dem).rename("slope_copdem_deg")
    slope_rad = slope_deg.multiply(math.pi / 180.0)

    # MERIT Hydro upstream drainage area, in km^2
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    upa_km2 = merit.select("upa")
    upa_m2 = upa_km2.multiply(1e6)

    # TWI proxy
    twi_proxy = (
        upa_m2.add(1.0)
        .divide(slope_rad.tan().add(1e-6))
        .log()
        .rename("twi_proxy")
    )

    return dem.addBands(slope_deg).addBands(twi_proxy)


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
    parser.add_argument("--csv", type=str, required=True, help="Path to city_grid_base_centroids_gaia_with_gee.csv")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    args = parser.parse_args()

    in_csv = Path(args.csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    out_csv = Path(args.out) if args.out else in_csv.with_name("city_grid_base_centroids_gaia_terrain_api.csv")

    df = pd.read_csv(in_csv)
    required = ["grid_id", "city_clean", "centroid_lon", "centroid_lat"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df["centroid_lon"] = pd.to_numeric(df["centroid_lon"], errors="coerce")
    df["centroid_lat"] = pd.to_numeric(df["centroid_lat"], errors="coerce")
    df = df.dropna(subset=["grid_id", "city_clean", "centroid_lon", "centroid_lat"]).copy()

    print(f"Rows to sample: {len(df)}")
    print("Variables: elevation_copdem_m, slope_copdem_deg, twi_proxy")

    init_ee(args.project)
    stack = build_stack()

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
