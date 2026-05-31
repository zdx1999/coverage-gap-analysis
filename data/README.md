# Data files expected by the code

Place analysis-ready input files in this directory before running the scripts.

## Required core tables

### `city_event_grid_full_gaia_v4_poi_repaired_253cities.csv`

Main city-event-grid table. Required columns include:

- `Event_ID`
- `city_clean`
- `grid_id`
- `centroid_lon`
- `centroid_lat`
- `cell_size_m`
- `flood_count`
- `is_extreme`
- `worldpop_2020`
- `poi_count_2018`
- `Event_Peak_Rain`

Optional columns used by some figure panels:

- `hotspot_refined`
- `new_hotspot_region`
- `first_flood_time`
- `last_flood_time`
- `dist_to_center_km`
- `ntl_avg_rad_2021`
- `elevation_copdem_m`
- `slope_copdem_deg`
- `twi_proxy`

### `city_grid_base_centroids_gaia_with_gee_terrain_repaired_253cities.csv`

Grid-level static baseline table. Required for road exposure and some mechanism models.

Required identifiers:

- `city_clean`
- `grid_id`

Road columns are inferred from available names, including examples such as:

- `road_len_km_grip4_l2`
- `road_len_km_grip4_l3`

### `grid_poi_baseline_2018_bytype.csv`

POI baseline by grid and function group. Required for some map/support-density panels.

### `Weibo_Flood_Master_V4_SpatialCleaned.csv`

Cleaned point-level flood-report table used for Fig. 1 map panels.

Required columns include:

- `clean_id`
- `clean_lon`
- `clean_lat`
- `Event_ID`
- `city_clean`

### `gdp2020.csv`

City-level GDP per capita table for Fig. 5 bar colours.

Required columns:

- `city_clean`
- `gdp_per_capita`

Alternatively, the script can compute `gdp_per_capita` when both `gdp_total` and `resident_population` are available.

## Optional geospatial inputs

Place shapefiles in `data/shapefiles/` using these filenames when reproducing map panels:

- `china_prefecture.shp`
- `china_province.shp`
- `china_national_border.shp`
- `nine_dash_line.shp`
- `south_china_islands.shp`

Fig. 1 can also use GAIA tiles if placed under:

```text
data/GAIA_2024_Data/
```

## Privacy and redistribution

Do not upload raw social-media text, private user identifiers, precise sensitive records, or third-party licensed geospatial datasets unless redistribution rights are clear. For public GitHub release, prefer sharing processed, de-identified derived tables or linking to an approved data repository.

