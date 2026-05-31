"""
Fig. 1 layout template v5 for manuscript:
"National study system and analytical framework of dynamic flood-exposure reconfiguration"

Layout v5
---------
Top row:
    a. Conceptual framing
    b. Data processing and analytical workflow

Bottom region:
    c. National extreme-event city sample
       - China map in the center
       - 2 city-event insets on the left
       - 2 city-event insets on the right
       - 6 city-event insets along the bottom

Important
---------
This script keeps the v5 manual layout and replaces the original placeholders
with the manuscript's repaired 253-city grid-event table and real municipal
boundary data where available.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
from pathlib import Path

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.windows import from_bounds
except Exception:
    rasterio = None
    Resampling = None
    geometry_mask = None
    transform_from_bounds = None
    from_bounds = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

try:
    from cartopy.io import shapereader
except Exception:
    shapereader = None

try:
    import contextily as cx
except Exception:
    cx = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SUPPORT_DIR = REPO_ROOT / "scripts" / "support"
for path in [SUPPORT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fig2a_national_footprint as nat
import fig2_zhengzhou_5551_final_refined_v5 as zz


# =========================================================
# Global settings
# =========================================================
plt.rcParams["font.family"] = "DejaVu Sans"  # Replace with Arial/Helvetica if available
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 8.5
plt.rcParams["ytick.labelsize"] = 8.5
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =========================================================
# Colors
# =========================================================
COLORS = {
    "map_bg": "#F1F3F3",
    "land": "#E3E3E3",
    "border": "#555555",
    "city_all": "#B8B8B8",
    "city_extreme": "#1B8A8F",
    "city_high": "#8C2D2D",
    "builtup": "#DADADA",
    "recurrent": "#8C2D2D",
    "new": "#1B8A8F",
    "function": "#D49A2A",
    "pipeline": "#EAF3F3",
    "pipeline_edge": "#3B777A",
    "text": "#222222",
    "gray_text": "#555555",
}

DEFAULT_FULL_TABLE = zz.DEFAULT_FULL_TABLE
DEFAULT_SHP = zz.DEFAULT_SHP
DEFAULT_POINT_TABLE = REPO_ROOT / "data" / "Weibo_Flood_Master_V4_SpatialCleaned.csv"
DEFAULT_GAIA_DIR = REPO_ROOT / "data" / "GAIA_2024_Data"
DEFAULT_PROVINCE_SHP = REPO_ROOT / "data" / "shapefiles" / "china_province.shp"
DEFAULT_PREFECTURE_SHP = REPO_ROOT / "data" / "shapefiles" / "china_prefecture.shp"
DEFAULT_NATIONAL_BORDER_SHP = REPO_ROOT / "data" / "shapefiles" / "china_national_border.shp"
DEFAULT_NINE_DASH_SHP = REPO_ROOT / "data" / "shapefiles" / "nine_dash_line.shp"
DEFAULT_SOUTH_CHINA_ISLANDS_SHP = REPO_ROOT / "data" / "shapefiles" / "south_china_islands.shp"

DEFAULT_PANEL_A_PNG = REPO_ROOT / "data" / "figure_inputs" / "fig1a_concept.png"
DEFAULT_PANEL_B_PNG = REPO_ROOT / "data" / "figure_inputs" / "fig1b_workflow.png"
PANEL_LETTER_SIZE = 13.5

CITY_INSETS = [
    # Zhengzhou and Nanjing positions are swapped while keeping c-labels tied to layout positions.
    ("Nanjing", "南京市", "c1"),
    ("Guangzhou", "广州市", "c2"),
    ("Beijing", "北京", "c3"),
    ("Shanghai", "上海", "c4"),
    ("Wuhan", "武汉市", "c5"),
    ("Chongqing", "重庆", "c6"),
    ("Chengdu", "成都市", "c7"),
    ("Zhengzhou", "郑州市", "c8"),
    ("Hangzhou", "杭州市", "c9"),
    ("Shenzhen", "深圳市", "c10"),
]


# =========================================================
# Basic helpers
# =========================================================
ALBERS_CRS = (
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
ALBERS_TRANSFORMER = (
    Transformer.from_crs("EPSG:4326", ALBERS_CRS, always_xy=True)
    if Transformer is not None else None
)
WEBMERCATOR_TRANSFORMER = (
    Transformer.from_crs(ALBERS_CRS, "EPSG:3857", always_xy=True)
    if Transformer is not None else None
)
# Slightly tighter extent for the central China map so the basemap appears larger.
MAIN_CHINA_BOUNDS = (78.0, 18.0, 134.6, 53.7)
SOUTH_CHINA_SEA_BOUNDS = (105, 2.2, 122.2, 24.6)

# Try the online terrain basemap again; fallback still works if it is slow or unavailable.
USE_WORLD_TERRAIN_BASEMAP = True
DRAW_INTERNAL_PREFECTURE_BOUNDARIES = False
BUILTUP_CMAP = ListedColormap([(0.50, 0.50, 0.50, 0.42)])
OCEAN_CMAP = LinearSegmentedColormap.from_list(
    "ocean_wash",
    ["#DCEBF1", "#EAF4F6", "#D2E5EE"],
)
NATURAL_EARTH_CACHE = None
FLOOD_KDE_CMAP = LinearSegmentedColormap.from_list(
    "flood_kde_red",
    ["#FFF4E8", "#F4A08A", "#C93E3A"],
)
PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
DEAD_PROXY_MARKERS = ("127.0.0.1:9", "localhost:9")


def to_albers(gdf):
    if gdf is None or getattr(gdf, "empty", True):
        return gdf
    try:
        return gdf.to_crs(ALBERS_CRS)
    except Exception:
        return gdf


def project_lonlat(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if ALBERS_TRANSFORMER is None:
        return lon, lat
    return ALBERS_TRANSFORMER.transform(lon, lat)


def project_bounds(
    bounds: tuple[float, float, float, float],
    samples: int = 32,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    xs = np.r_[
        np.linspace(xmin, xmax, samples),
        np.linspace(xmin, xmax, samples),
        np.full(samples, xmin),
        np.full(samples, xmax),
    ]
    ys = np.r_[
        np.full(samples, ymin),
        np.full(samples, ymax),
        np.linspace(ymin, ymax, samples),
        np.linspace(ymin, ymax, samples),
    ]
    px, py = project_lonlat(xs, ys)
    return (
        float(np.nanmin(px)),
        float(np.nanmin(py)),
        float(np.nanmax(px)),
        float(np.nanmax(py)),
    )


def projected_grid(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
    north_to_south: bool = False,
):
    lon_edges = np.linspace(xmin, xmax, width + 1)
    lat_edges = np.linspace(ymax, ymin, height + 1) if north_to_south else np.linspace(ymin, ymax, height + 1)
    lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges)
    return project_lonlat(lon_grid, lat_grid)


def set_projected_extent(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    pad_frac: float = 0.0,
) -> None:
    pxmin, pymin, pxmax, pymax = project_bounds(bounds)
    pad_x = (pxmax - pxmin) * pad_frac
    pad_y = (pymax - pymin) * pad_frac
    ax.set_xlim(pxmin - pad_x, pxmax + pad_x)
    ax.set_ylim(pymin - pad_y, pymax + pad_y)


def draw_ocean_basemap(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
) -> None:
    xmin, ymin, xmax, ymax = bounds
    nx, ny = 80, 58
    gx, gy = projected_grid(xmin, ymin, xmax, ymax, nx, ny)
    xx = np.linspace(0, 1, nx)
    yy = np.linspace(0, 1, ny)
    wash = 0.42 + 0.28 * yy[:, None] + 0.08 * np.cos(xx[None, :] * np.pi)
    ax.pcolormesh(
        gx,
        gy,
        wash,
        cmap=OCEAN_CMAP,
        alpha=0.82,
        shading="flat",
        linewidth=0,
        edgecolors="none",
        rasterized=True,
        zorder=0,
    )
    for lon in np.arange(80, 126, 10):
        lat_line = np.linspace(ymin, ymax, 180)
        lon_line = np.full_like(lat_line, lon, dtype=float)
        px, py = project_lonlat(lon_line, lat_line)
        ax.plot(px, py, color="white", linewidth=0.34, alpha=0.24, zorder=0.5)
    for lat in np.arange(20, 55, 10):
        lon_line = np.linspace(xmin, xmax, 220)
        lat_line = np.full_like(lon_line, lat, dtype=float)
        px, py = project_lonlat(lon_line, lat_line)
        ax.plot(px, py, color="white", linewidth=0.34, alpha=0.24, zorder=0.5)


def load_natural_earth_layers() -> dict:
    global NATURAL_EARTH_CACHE
    if NATURAL_EARTH_CACHE is not None:
        return NATURAL_EARTH_CACHE
    layers = {"land": None, "ocean": None, "borders": None}
    if nat.gpd is None or shapereader is None:
        NATURAL_EARTH_CACHE = layers
        return layers
    sources = {
        "land": ("physical", "land"),
        "ocean": ("physical", "ocean"),
        "borders": ("cultural", "admin_0_boundary_lines_land"),
    }
    for key, (category, name) in sources.items():
        try:
            path = shapereader.natural_earth(
                resolution="50m",
                category=category,
                name=name,
            )
            layers[key] = nat.gpd.read_file(path).to_crs(4326)
        except Exception:
            layers[key] = None
    NATURAL_EARTH_CACHE = layers
    return layers


def subset_layer(gdf, bounds: tuple[float, float, float, float], pad: float = 1.5):
    if gdf is None or getattr(gdf, "empty", True):
        return None
    xmin, ymin, xmax, ymax = bounds
    try:
        return gdf.cx[xmin - pad:xmax + pad, ymin - pad:ymax + pad].copy()
    except Exception:
        return gdf


def draw_natural_earth_basemap(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    include_land: bool = True,
    include_borders: bool = True,
) -> None:
    draw_ocean_basemap(ax, bounds)
    layers = load_natural_earth_layers()
    ocean = to_albers(subset_layer(layers.get("ocean"), bounds, pad=4.0))
    land = to_albers(subset_layer(layers.get("land"), bounds, pad=2.0)) if include_land else None
    borders = to_albers(subset_layer(layers.get("borders"), bounds, pad=2.0)) if include_borders else None

    if ocean is not None and not ocean.empty:
        ocean.plot(ax=ax, facecolor="#D8EAF1", edgecolor="none", alpha=0.92, zorder=0.2)
    if land is not None and not land.empty:
        land.plot(ax=ax, facecolor="#ECE3D0", edgecolor="#A59D8D", linewidth=0.28, alpha=0.98, zorder=1.0)
        land.boundary.plot(ax=ax, edgecolor="#6F6A60", linewidth=0.36, alpha=0.75, zorder=1.2)
    if borders is not None and not borders.empty:
        borders.plot(ax=ax, color="#817B70", linewidth=0.28, alpha=0.60, zorder=1.3)


def clear_dead_proxy_env() -> dict[str, str]:
    """Temporarily remove black-hole proxy settings that block online map tiles."""
    removed = {}
    for key in PROXY_ENV_VARS:
        value = os.environ.get(key)
        if value and any(marker in value for marker in DEAD_PROXY_MARKERS):
            removed[key] = value
            os.environ.pop(key, None)
    return removed


def restore_proxy_env(removed: dict[str, str]) -> None:
    for key, value in removed.items():
        os.environ[key] = value


def current_axis_bounds_3857(ax: plt.Axes, samples: int = 18):
    if WEBMERCATOR_TRANSFORMER is None:
        return None
    xmin, xmax, ymin, ymax = ax.axis()
    xs = np.r_[
        np.linspace(xmin, xmax, samples),
        np.linspace(xmin, xmax, samples),
        np.full(samples, xmin),
        np.full(samples, xmax),
    ]
    ys = np.r_[
        np.full(samples, ymin),
        np.full(samples, ymax),
        np.linspace(ymin, ymax, samples),
        np.linspace(ymin, ymax, samples),
    ]
    mx, my = WEBMERCATOR_TRANSFORMER.transform(xs, ys)
    return float(np.nanmin(mx)), float(np.nanmin(my)), float(np.nanmax(mx)), float(np.nanmax(my))


def draw_worldterrain_basemap(
    ax: plt.Axes,
    zoom: int = 4,
    alpha: float = 0.90,
    timeout_s: float = 6.0,
) -> bool:
    """Draw an online terrain basemap under the projected China layers."""
    if not USE_WORLD_TERRAIN_BASEMAP:
        return False
    if cx is None or WEBMERCATOR_TRANSFORMER is None or Resampling is None:
        return False
    old_xlim = ax.get_xlim()
    old_ylim = ax.get_ylim()
    removed_proxy = clear_dead_proxy_env()
    import requests
    original_get = requests.get

    def get_with_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", timeout_s)
        return original_get(*args, **kwargs)

    requests.get = get_with_timeout
    try:
        wm_bounds = current_axis_bounds_3857(ax)
        if wm_bounds is None:
            return False
        image, extent = cx.bounds2img(
            *wm_bounds,
            zoom=zoom,
            source=cx.providers.Esri.WorldTerrain,
            ll=False,
            max_retries=0,
            n_connections=1,
            use_cache=True,
        )
        image, extent = cx.warp_tiles(
            image,
            extent,
            t_crs=ALBERS_CRS,
            resampling=Resampling.bilinear,
        )
        if image.shape[2] == 1:
            image = image[:, :, 0]
        ax.imshow(
            image,
            extent=extent,
            interpolation="bilinear",
            aspect=ax.get_aspect(),
            alpha=alpha,
            zorder=0.4,
        )
        ax.set_xlim(old_xlim)
        ax.set_ylim(old_ylim)
        return True
    except Exception as exc:
        print(f"WorldTerrain basemap unavailable; using local fallback. Reason: {exc}", file=sys.stderr)
        progress(f"WorldTerrain basemap unavailable: {exc}")
        ax.set_xlim(old_xlim)
        ax.set_ylim(old_ylim)
        return False
    finally:
        requests.get = original_get
        restore_proxy_env(removed_proxy)


def load_admin_basemap(
    province_path: str | Path = DEFAULT_PROVINCE_SHP,
    prefecture_path: str | Path = DEFAULT_PREFECTURE_SHP,
) -> dict:
    layers = {"province": None, "prefecture": None}
    if nat.gpd is None:
        return layers
    for key, path in [("province", province_path), ("prefecture", prefecture_path)]:
        path = Path(path)
        if not path.exists():
            continue
        try:
            gdf = nat.gpd.read_file(path)
            if gdf.crs is None:
                gdf = gdf.set_crs(4326, allow_override=True)
            gdf = gdf.to_crs(4326)
            if key == "prefecture" and "city_std" not in gdf.columns:
                cname = nat.infer_city_name_col(gdf)
                if cname is not None:
                    gdf["city_std"] = gdf[cname].map(nat.standardize_city_name)
            layers[key] = gdf
        except Exception:
            layers[key] = None
    return layers


def load_national_border(shp_path: str | Path = DEFAULT_NATIONAL_BORDER_SHP):
    shp_path = Path(shp_path)
    if nat.gpd is None or not shp_path.exists():
        return None
    try:
        gdf = nat.gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326, allow_override=True)
        return gdf.to_crs(4326)
    except Exception:
        return None


def load_nine_dash_line(shp_path: str | Path = DEFAULT_NINE_DASH_SHP):
    shp_path = Path(shp_path)
    if nat.gpd is None or not shp_path.exists():
        return None
    try:
        gdf = nat.gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326, allow_override=True)
        return gdf.to_crs(ALBERS_CRS)
    except Exception:
        return None


def load_south_china_islands(shp_path: str | Path = DEFAULT_SOUTH_CHINA_ISLANDS_SHP):
    shp_path = Path(shp_path)
    if nat.gpd is None or not shp_path.exists():
        return None
    try:
        gdf = nat.gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326, allow_override=True)
        return gdf.to_crs(ALBERS_CRS)
    except Exception:
        return None


def draw_nine_dash_line(ax: plt.Axes, nine_dash=None, linewidth: float = 0.95) -> None:
    if nine_dash is None or getattr(nine_dash, "empty", True):
        return
    nine_dash.plot(
        ax=ax,
        color="#4F4F4F",
        linewidth=linewidth,
        alpha=0.92,
        zorder=5,
    )


def draw_south_china_islands(ax: plt.Axes, islands=None, linewidth: float = 0.42) -> None:
    if islands is None or getattr(islands, "empty", True):
        return
    islands.plot(
        ax=ax,
        color="#5A5A5A",
        linewidth=linewidth,
        alpha=0.86,
        zorder=4.8,
    )


def draw_admin_overlay(
    ax: plt.Axes,
    admin: dict | None,
    bounds: tuple[float, float, float, float],
    inset: bool = False,
) -> None:
    if not admin:
        return
    province = to_albers(subset_layer(admin.get("province"), bounds, pad=0.8))
    prefecture = to_albers(subset_layer(admin.get("prefecture"), bounds, pad=0.8))
    if province is not None and not province.empty:
        province.plot(
            ax=ax,
            facecolor="#F0E5CF",
            edgecolor="none",
            alpha=0.18 if not inset else 0.12,
            zorder=1.45,
        )
    if prefecture is not None and not prefecture.empty:
        prefecture.boundary.plot(
            ax=ax,
            edgecolor="#A6A6A6",
            linewidth=0.08 if not inset else 0.07,
            alpha=0.55 if not inset else 0.48,
            zorder=1.68,
        )
    if province is not None and not province.empty:
        province.boundary.plot(
            ax=ax,
            edgecolor="#6A6A6A",
            linewidth=0.22 if not inset else 0.18,
            alpha=0.82,
            zorder=1.88,
        )


def draw_china_city_boundary_overlay(
    ax: plt.Axes,
    national: dict,
    bounds: tuple[float, float, float, float],
) -> None:
    """Draw city, province, and national boundaries with a clear visual hierarchy, plus the ten inset-city boundaries."""
    progress("national map overlay: load national border")
    border = national.get("national_border") if national else None
    border = to_albers(subset_layer(border, bounds, pad=0.8)) if border is not None else None

    progress("national map overlay: load province/prefecture layers")
    admin = national.get("admin") if national else None
    province = to_albers(subset_layer(admin.get("province"), bounds, pad=0.8)) if admin and admin.get("province") is not None else None
    prefecture = to_albers(subset_layer(admin.get("prefecture"), bounds, pad=0.8)) if admin and admin.get("prefecture") is not None else None

    if prefecture is not None and not prefecture.empty:
        progress("national map overlay: draw city boundaries")
        prefecture.boundary.plot(
            ax=ax,
            edgecolor="#A7A7A7",
            linewidth=0.08,
            alpha=0.52,
            zorder=2.02,
        )
    else:
        progress("national map overlay: city boundary layer unavailable")

    if province is not None and not province.empty:
        progress("national map overlay: draw province boundaries")
        province.boundary.plot(
            ax=ax,
            edgecolor="#666666",
            linewidth=0.22,
            alpha=0.84,
            zorder=2.35,
        )
    else:
        progress("national map overlay: province boundary layer unavailable")

    if border is not None and not border.empty:
        progress("national map overlay: draw national border shapefile")
        try:
            border.boundary.plot(
                ax=ax,
                edgecolor="#1E1E1E",
                linewidth=0.72,
                alpha=0.96,
                zorder=3.20,
            )
        except Exception:
            progress("national map overlay: national border draw failed; fallback skipped")
    else:
        progress("national map overlay: national border shapefile not found")

    inset_city_stds = {
        nat.standardize_city_name(city_name)
        for _, city_name, _ in CITY_INSETS
    }
    progress("national map overlay: select 10 inset-city boundaries")
    inset_boundaries = national.get("shp")
    if inset_boundaries is not None and not inset_boundaries.empty:
        inset_boundaries = inset_boundaries[inset_boundaries["city_std"].isin(inset_city_stds)].copy()
        inset_boundaries = to_albers(inset_boundaries)
        if inset_boundaries is not None and not inset_boundaries.empty:
            progress("national map overlay: fill inset-city footprints")
            inset_boundaries.plot(
                ax=ax,
                facecolor="#FFE79A",
                edgecolor="none",
                alpha=0.26,
                zorder=3.35,
            )
            progress("national map overlay: draw inset-city boundaries")
            inset_boundaries.boundary.plot(
                ax=ax,
                edgecolor="#C7771E",
                linewidth=0.86,
                alpha=0.98,
                zorder=3.65,
            )


def style_panel(ax: plt.Axes, facecolor: str = "white", border: bool = True) -> None:
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    if border:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#333333")
            spine.set_zorder(20)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)


def reinforce_axes_frame(ax: plt.Axes, linewidth: float = 1.2, color: str = "#333333", zorder: float = 30.0) -> None:
    """Redraw the full panel frame to keep the top/right border visible after heavy map layers."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_color(color)
        spine.set_zorder(zorder)
    ax.add_patch(
        Rectangle(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            zorder=zorder,
            clip_on=False,
        )
    )


def progress(message: str) -> None:
    print(f"[fig1] {message}", flush=True)


def draw_placeholder_china(ax: plt.Axes) -> None:
    """
    Placeholder stylized China-like map.

    Replace with real China boundary later:
        - national boundary
        - province boundaries if needed
        - optional elevation/hillshade/neutral basemap
        - real projection
    """
    ax.set_xlim(73, 136)
    ax.set_ylim(17, 54)
    ax.set_aspect("equal")
    ax.set_facecolor(COLORS["map_bg"])

    poly_coords = np.array([
        [75, 39], [79, 45], [87, 49], [96, 49], [104, 53],
        [116, 50], [123, 46], [132, 47], [134, 42], [127, 39],
        [123, 35], [121, 30], [116, 27], [112, 22], [105, 21],
        [100, 24], [94, 23], [88, 27], [83, 31], [78, 34]
    ])

    ax.add_patch(
        Polygon(
            poly_coords,
            closed=True,
            facecolor=COLORS["land"],
            edgecolor=COLORS["border"],
            linewidth=1.2,
            zorder=1
        )
    )

    # Stylized rivers / guides
    x = np.linspace(82, 123, 200)
    y = 32 + 2.2 * np.sin((x - 82) / 8)
    ax.plot(x, y, color="white", linewidth=1.2, alpha=0.9, zorder=2)

    x2 = np.linspace(96, 124, 120)
    y2 = 39 + 1.5 * np.sin((x2 - 96) / 7)
    ax.plot(x2, y2, color="white", linewidth=1.0, alpha=0.85, zorder=2)


def draw_city_points(ax: plt.Axes, seed: int = 11) -> None:
    """
    Placeholder national city points.

    Replace with real city coordinates later.
    Recommended real columns:
        city_id, city_name, lon, lat,
        is_analytical_city, has_extreme_event,
        extreme_event_count, flood_report_count,
        newly_opened_exposure_burden
    """
    rng = np.random.default_rng(seed)

    n_all = 253
    lon = rng.uniform(80, 126, n_all)
    lat = rng.uniform(22, 48, n_all)

    keep = ((lon - 101) / 27) ** 2 + ((lat - 35) / 16) ** 2 < 1.05
    lon = lon[keep]
    lat = lat[keep]

    while len(lon) < n_all:
        extra_lon = rng.uniform(80, 126, n_all)
        extra_lat = rng.uniform(22, 48, n_all)
        keep = ((extra_lon - 101) / 27) ** 2 + ((extra_lat - 35) / 16) ** 2 < 1.05
        lon = np.r_[lon, extra_lon[keep]]
        lat = np.r_[lat, extra_lat[keep]]

    lon = lon[:n_all]
    lat = lat[:n_all]

    extreme_idx = rng.choice(np.arange(n_all), size=180, replace=False)
    high_idx = rng.choice(extreme_idx, size=28, replace=False)

    event_counts = rng.integers(1, 12, n_all)
    sizes = 8 + event_counts * 3

    ax.scatter(
        lon, lat, s=10, color=COLORS["city_all"],
        edgecolor="white", linewidth=0.3, alpha=0.72, zorder=3
    )

    ax.scatter(
        lon[extreme_idx], lat[extreme_idx], s=sizes[extreme_idx],
        color=COLORS["city_extreme"], edgecolor="white",
        linewidth=0.35, alpha=0.88, zorder=4
    )

    ax.scatter(
        lon[high_idx], lat[high_idx], s=sizes[high_idx] + 22,
        color=COLORS["city_high"], edgecolor="white",
        linewidth=0.45, alpha=0.95, zorder=5
    )


def load_repaired_grid_table(full_table: str | Path = DEFAULT_FULL_TABLE) -> pd.DataFrame:
    """Load the columns needed for the national map and representative insets."""
    full_table = Path(full_table)
    needed = [
        "Event_ID",
        "city_clean",
        "grid_id",
        "centroid_lon",
        "centroid_lat",
        "cell_size_m",
        "flood_count",
        "is_extreme",
        "hotspot_refined",
        "new_hotspot_region",
    ]
    header = pd.read_csv(full_table, nrows=0, low_memory=False)
    usecols = [c for c in needed if c in header.columns]
    df = pd.read_csv(full_table, usecols=usecols, low_memory=False)
    for c in ["Event_ID", "centroid_lon", "centroid_lat", "cell_size_m", "flood_count", "is_extreme", "hotspot_refined", "new_hotspot_region"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "city_clean" not in df.columns:
        raise ValueError("city_clean column not found in the repaired grid-event table.")
    df["city_std"] = df["city_clean"].map(nat.standardize_city_name)
    return df


def load_flood_points(point_table: str | Path = DEFAULT_POINT_TABLE) -> pd.DataFrame:
    """Load event-level Weibo-derived flood-report points for Fig. 1c insets."""
    point_table = Path(point_table)
    if not point_table.exists():
        return pd.DataFrame()

    header = pd.read_csv(point_table, nrows=0, low_memory=False)
    preferred = ["clean_id", "clean_lon", "clean_lat", "city_clean", "城市", "Event_ID", "clean_event_id"]
    usecols = [c for c in preferred if c in header.columns]
    pts = pd.read_csv(point_table, usecols=usecols, low_memory=False)

    lon_col = "clean_lon" if "clean_lon" in pts.columns else None
    lat_col = "clean_lat" if "clean_lat" in pts.columns else None
    event_col = "Event_ID" if "Event_ID" in pts.columns else "clean_event_id" if "clean_event_id" in pts.columns else None
    if lon_col is None or lat_col is None or event_col is None:
        return pd.DataFrame()

    city_col = "city_clean" if "city_clean" in pts.columns else "城市" if "城市" in pts.columns else None
    pts = pts.rename(columns={lon_col: "lon", lat_col: "lat", event_col: "Event_ID"})
    pts["lon"] = pd.to_numeric(pts["lon"], errors="coerce")
    pts["lat"] = pd.to_numeric(pts["lat"], errors="coerce")
    pts["Event_ID"] = pd.to_numeric(pts["Event_ID"], errors="coerce")
    pts = pts[pts["lon"].between(73, 136) & pts["lat"].between(17, 54) & pts["Event_ID"].notna()].copy()
    if city_col is not None:
        pts["city_std"] = pts[city_col].map(nat.standardize_city_name)
    else:
        pts["city_std"] = ""
    return pts


def prepare_national_sample(df: pd.DataFrame, shp_path: str | Path = DEFAULT_SHP) -> dict | None:
    """Prepare real municipal boundaries and city-level sample summaries."""
    shp_path = Path(shp_path)
    if nat.gpd is None or not shp_path.exists():
        return None

    event = (
        df.groupby(["city_clean", "city_std", "Event_ID", "is_extreme"], as_index=False)
        .agg(
            event_burden=("flood_count", "sum"),
            hotspot_grids=("hotspot_refined", "sum"),
            new_hotspot_grids=("new_hotspot_region", "sum"),
        )
    )
    city_all = event.groupby(["city_clean", "city_std"], as_index=False).agg(
        total_events=("Event_ID", "nunique"),
        total_reports=("event_burden", "sum"),
    )
    city_extreme = (
        event[event["is_extreme"] == 1]
        .groupby(["city_clean", "city_std"], as_index=False)
        .agg(
            extreme_events=("Event_ID", "nunique"),
            extreme_reports=("event_burden", "sum"),
            hotspot_grids=("hotspot_grids", "sum"),
            new_hotspot_grids=("new_hotspot_grids", "sum"),
        )
    )

    shp = nat.gpd.read_file(shp_path)
    cname = nat.infer_city_name_col(shp)
    if cname is None:
        return None
    shp["city_std"] = shp[cname].map(nat.standardize_city_name)
    if shp.crs is None:
        shp = shp.set_crs(4326, allow_override=True)
    shp = shp.to_crs(4326)

    universe = shp[shp["city_std"].isin(city_all["city_std"])].copy()
    universe = universe.merge(city_all, on="city_std", how="left")
    extreme = shp.merge(city_extreme, on="city_std", how="inner")

    for gdf in [universe, extreme]:
        pts = gdf.representative_point()
        gdf["rep_x"] = pts.x
        gdf["rep_y"] = pts.y

    high_n = min(24, len(extreme))
    high = extreme.sort_values(["new_hotspot_grids", "hotspot_grids", "extreme_reports"], ascending=False).head(high_n).copy()
    admin = load_admin_basemap()
    national_border = load_national_border()

    return {
        "shp": shp,
        "admin": admin,
        "national_border": national_border,
        "universe": universe,
        "extreme": extreme,
        "high": high,
        "n_all": int(city_all["city_std"].nunique()),
        "n_extreme_cities": int(city_extreme["city_std"].nunique()),
        "n_extreme_events": int(event[event["is_extreme"] == 1]["Event_ID"].nunique()),
        "n_grid_events": int(len(df[df["is_extreme"] == 1])),
        "n_hotspots": int(pd.to_numeric(df.loc[df["is_extreme"] == 1, "hotspot_refined"], errors="coerce").fillna(0).sum()),
    }


def draw_national_city_markers(ax: plt.Axes, national: dict | None) -> None:
    """Differentiate analytical cities, extreme-event cities, and the 10 inset cities on the national map."""
    if not national:
        return

    progress("national map: draw city markers")
    universe = national.get("universe")
    extreme = national.get("extreme")
    inset_city_stds = {
        nat.standardize_city_name(city_name)
        for _, city_name, _ in CITY_INSETS
    }

    if universe is not None and not universe.empty:
        ux, uy = project_lonlat(universe["rep_x"], universe["rep_y"])
        ax.scatter(
            ux, uy,
            s=8,
            facecolors="#D4D4D4",
            edgecolors="white",
            linewidths=0.18,
            alpha=0.75,
            zorder=4.00,
        )

    if extreme is not None and not extreme.empty:
        ex, ey = project_lonlat(extreme["rep_x"], extreme["rep_y"])
        ax.scatter(
            ex, ey,
            s=15,
            facecolors="#5B86B0",
            edgecolors="white",
            linewidths=0.24,
            alpha=0.88,
            zorder=4.35,
        )

        inset_pts = extreme[extreme["city_std"].isin(inset_city_stds)].copy()
    else:
        inset_pts = None

    if inset_pts is None or inset_pts.empty:
        if universe is not None and not universe.empty:
            inset_pts = universe[universe["city_std"].isin(inset_city_stds)].copy()


def draw_real_china_map(ax: plt.Axes, national: dict | None, nine_dash=None, islands=None) -> None:
    """Draw the real national analytical sample; fall back to placeholders if needed."""
    if national is None:
        progress("national map: fallback placeholder")
        draw_placeholder_china(ax)
        draw_city_points(ax)
        return

    progress("national map: set extent")
    set_projected_extent(ax, MAIN_CHINA_BOUNDS, pad_frac=0.004)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#DCEBF1")

    progress("national map: try online terrain basemap")
    basemap_drawn = draw_worldterrain_basemap(ax)
    if basemap_drawn:
        progress("national map: online terrain basemap loaded")
    else:
        progress("national map: online terrain basemap unavailable; use local fallback")
        draw_ocean_basemap(ax, MAIN_CHINA_BOUNDS)

    progress("national map: draw China land fill")
    fallback_fill = to_albers(subset_layer(national.get("shp"), MAIN_CHINA_BOUNDS, pad=0.8))
    if fallback_fill is not None and not fallback_fill.empty:
        # Keep land fill light enough so the terrain basemap remains visible when it loads.
        fill_alpha = 0.28 if basemap_drawn else 0.62
        fallback_fill.plot(ax=ax, facecolor="#EFE6D0", edgecolor="none", alpha=fill_alpha, zorder=0.9)

    progress("national map: draw China outline + 10 inset-city boundaries")
    draw_china_city_boundary_overlay(ax, national, MAIN_CHINA_BOUNDS)

    draw_national_city_markers(ax, national)

    progress("national map: draw nine-dash line on main map")
    draw_nine_dash_line(ax, nine_dash=nine_dash, linewidth=0.82)

    progress("national map: draw South China Sea inset")
    draw_south_china_sea_inset(ax, national.get("admin"), nine_dash, islands)
    progress("national map: reinforce main map frame")
    reinforce_axes_frame(ax, linewidth=1.25, color="#333333")
    progress("national map: done")


def draw_south_china_sea_inset(parent_ax: plt.Axes, admin=None, nine_dash=None, islands=None) -> None:
    """Move the South China Sea extent into a compact inset."""
    progress("South China Sea inset: create axes")
    ax = parent_ax.inset_axes([0.775, 0.052, 0.200, 0.255], zorder=9)
    style_panel(ax, facecolor="#DCEBF1", border=True)
    progress("South China Sea inset: set extent")
    set_projected_extent(ax, SOUTH_CHINA_SEA_BOUNDS, pad_frac=0.025)
    progress("South China Sea inset: draw background")
    if not draw_worldterrain_basemap(ax, zoom=3, alpha=0.82, timeout_s=4.0):
        draw_natural_earth_basemap(ax, SOUTH_CHINA_SEA_BOUNDS, include_land=False, include_borders=False)
    progress("South China Sea inset: draw overlays")
    draw_admin_overlay(ax, admin, SOUTH_CHINA_SEA_BOUNDS, inset=True)
    draw_south_china_islands(ax, islands=islands, linewidth=0.40)
    draw_nine_dash_line(ax, nine_dash=nine_dash, linewidth=1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    progress("South China Sea inset: reinforce inset frame")
    reinforce_axes_frame(ax, linewidth=1.1, color="#333333")


def gaia_tiles_for_bounds(bounds: tuple[float, float, float, float], gaia_dir: str | Path = DEFAULT_GAIA_DIR) -> list[Path]:
    """Return 5-degree GAIA tiles intersecting a lon/lat bbox."""
    gaia_dir = Path(gaia_dir)
    if not gaia_dir.exists():
        return []
    xmin, ymin, xmax, ymax = bounds
    tiles = []
    for tif in gaia_dir.glob("GAIA_2024_*.tif"):
        parts = tif.stem.split("_")
        if len(parts) < 4:
            continue
        try:
            lon0 = float(parts[-2])
            lat0 = float(parts[-1])
        except ValueError:
            continue
        tile_xmin, tile_xmax = lon0, lon0 + 5
        # GAIA filenames use the northern latitude edge: e.g. *_110_35 spans 30-35N.
        tile_ymin, tile_ymax = lat0 - 5, lat0
        if tile_xmin <= xmax and tile_xmax >= xmin and tile_ymin <= ymax and tile_ymax >= ymin:
            tiles.append(tif)
    return sorted(tiles)


def draw_gaia_builtup(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    clip_boundary=None,
    gaia_dir: str | Path = DEFAULT_GAIA_DIR,
    project_to_albers: bool = False,
) -> bool:
    """Draw a legible 30 m GAIA built-up basemap, downsampled only for display."""
    if rasterio is None or from_bounds is None or Resampling is None:
        return False

    xmin, ymin, xmax, ymax = bounds
    clip_shapes = None
    if clip_boundary is not None and geometry_mask is not None and transform_from_bounds is not None:
        try:
            clip_shapes = [
                geom.__geo_interface__
                for geom in clip_boundary.geometry
                if geom is not None and not geom.is_empty
            ]
        except Exception:
            clip_shapes = None

    drawn = False
    for tif in gaia_tiles_for_bounds(bounds, gaia_dir):
        try:
            with rasterio.open(tif) as src:
                ixmin = max(xmin, src.bounds.left)
                ixmax = min(xmax, src.bounds.right)
                iymin = max(ymin, src.bounds.bottom)
                iymax = min(ymax, src.bounds.top)
                if ixmin >= ixmax or iymin >= iymax:
                    continue
                window = from_bounds(ixmin, iymin, ixmax, iymax, src.transform)
                window = window.round_offsets().round_lengths()
                height = max(1, int(window.height))
                width = max(1, int(window.width))
                scale = max(width / 560, height / 560, 1.0)
                out_h = max(1, int(height / scale))
                out_w = max(1, int(width / scale))
                arr = src.read(
                    1,
                    window=window,
                    out_shape=(out_h, out_w),
                    resampling=getattr(Resampling, "average", Resampling.nearest),
                )
                built = arr > 0.035
                if clip_shapes:
                    image_transform = transform_from_bounds(ixmin, iymin, ixmax, iymax, out_w, out_h)
                    inside_city = geometry_mask(
                        clip_shapes,
                        out_shape=built.shape,
                        transform=image_transform,
                        invert=True,
                    )
                    built = built & inside_city
                if not built.any():
                    continue

                if project_to_albers and ALBERS_TRANSFORMER is not None:
                    gx, gy = projected_grid(
                        ixmin, iymin, ixmax, iymax,
                        out_w, out_h,
                        north_to_south=True,
                    )
                    built_masked = np.ma.masked_where(~built, np.ones_like(built, dtype=float))
                    ax.pcolormesh(
                        gx,
                        gy,
                        built_masked,
                        cmap=BUILTUP_CMAP,
                        vmin=0,
                        vmax=1,
                        shading="flat",
                        linewidth=0,
                        edgecolors="none",
                        rasterized=True,
                        zorder=2,
                    )
                else:
                    rgba = np.zeros((built.shape[0], built.shape[1], 4), dtype=float)
                    rgba[..., 0][built] = 0.50
                    rgba[..., 1][built] = 0.50
                    rgba[..., 2][built] = 0.50
                    rgba[..., 3][built] = 0.42
                    ax.imshow(
                        rgba,
                        extent=(ixmin, ixmax, iymin, iymax),
                        origin="upper",
                        interpolation="nearest",
                        zorder=2,
                    )
                drawn = True
        except Exception:
            continue
    return drawn


def get_city_boundary(national: dict | None, city_name: str):
    if national is None or "shp" not in national:
        return None
    city_std = nat.standardize_city_name(city_name)
    gdf = national["shp"][national["shp"]["city_std"] == city_std].copy()
    return gdf if not gdf.empty else None


def city_points_for_inset(point_df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    """Return all flood-report points assigned to one inset city."""
    if point_df.empty:
        return pd.DataFrame()
    city_std = nat.standardize_city_name(city_name)
    return point_df[point_df["city_std"] == city_std].copy()


def draw_flood_point_kde(
    ax: plt.Axes,
    pts: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    clip_boundary=None,
    bins: int = 90,
    project_to_albers: bool = False,
) -> None:
    """Draw a pale binned KDE from all city flood-report points."""
    if pts.empty:
        return
    xmin, ymin, xmax, ymax = bounds
    visible = pts[
        pts["lon"].between(xmin, xmax)
        & pts["lat"].between(ymin, ymax)
    ].copy()
    if len(visible) < 5:
        return

    hist, _, _ = np.histogram2d(
        visible["lon"],
        visible["lat"],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
    )
    density = hist.T
    if gaussian_filter is not None:
        density = gaussian_filter(density, sigma=1.85)
    if not np.any(density > 0):
        return

    if clip_boundary is not None and geometry_mask is not None and transform_from_bounds is not None:
        try:
            shapes = [
                geom.__geo_interface__
                for geom in clip_boundary.geometry
                if geom is not None and not geom.is_empty
            ]
            if shapes:
                image_transform = transform_from_bounds(xmin, ymin, xmax, ymax, bins, bins)
                inside_city = geometry_mask(
                    shapes,
                    out_shape=density.shape,
                    transform=image_transform,
                    invert=True,
                )
                density = np.where(np.flipud(inside_city), density, np.nan)
        except Exception:
            pass

    positive = density[np.isfinite(density) & (density > 0)]
    if positive.size == 0:
        return
    floor = np.percentile(positive, 20)
    ceiling = np.percentile(positive, 97)
    if ceiling <= floor:
        ceiling = positive.max()
    density = np.ma.masked_where(~np.isfinite(density) | (density <= floor), density)
    if project_to_albers and ALBERS_TRANSFORMER is not None:
        gx, gy = projected_grid(xmin, ymin, xmax, ymax, bins, bins)
        ax.pcolormesh(
            gx,
            gy,
            density,
            cmap=FLOOD_KDE_CMAP,
            alpha=0.34,
            shading="flat",
            linewidth=0,
            edgecolors="none",
            rasterized=True,
            vmin=floor,
            vmax=ceiling,
            zorder=4,
        )
    else:
        ax.imshow(
            density,
            extent=(xmin, xmax, ymin, ymax),
            origin="lower",
            cmap=FLOOD_KDE_CMAP,
            alpha=0.34,
            interpolation="bilinear",
            vmin=floor,
            vmax=ceiling,
            zorder=4,
        )


def inset_lonlat_bounds(
    boundary,
    event_df: pd.DataFrame | None,
    point_df: pd.DataFrame,
    min_span: float = 0.18,
) -> tuple[float, float, float, float]:
    if boundary is not None and not boundary.empty:
        xmin, ymin, xmax, ymax = boundary.total_bounds
    else:
        xs = []
        ys = []
        if event_df is not None and not event_df.empty:
            xs.extend(pd.to_numeric(event_df["centroid_lon"], errors="coerce").dropna().tolist())
            ys.extend(pd.to_numeric(event_df["centroid_lat"], errors="coerce").dropna().tolist())
        if not point_df.empty:
            xs.extend(pd.to_numeric(point_df["lon"], errors="coerce").dropna().tolist())
            ys.extend(pd.to_numeric(point_df["lat"], errors="coerce").dropna().tolist())
        if not xs or not ys:
            return (110, 30, 111, 31)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

    span_x = max(xmax - xmin, min_span)
    span_y = max(ymax - ymin, min_span)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    side = max(span_x, span_y)
    # Use a very small padding so the city footprint fills each inset frame as much as possible.
    pad = side * 0.012
    side = side + 2 * pad
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def draw_real_city_inset(
    ax: plt.Axes,
    title: str,
    city_name: str,
    letter: str,
    point_df: pd.DataFrame | None = None,
    boundary=None,
    seed: int = 0,
) -> None:
    """Draw a data-layer city inset: boundary, GAIA built-up, all flood points and KDE."""
    pts = city_points_for_inset(point_df if point_df is not None else pd.DataFrame(), city_name)
    if pts.empty and (boundary is None or boundary.empty):
        draw_city_inset(ax, title, letter, seed=seed)
        return

    style_panel(ax, facecolor="none", border=False)

    bounds = inset_lonlat_bounds(boundary, None, pts)
    proj_bounds = project_bounds(bounds)
    boundary_proj = None
    if boundary is not None and not boundary.empty:
        boundary_proj = to_albers(boundary)
        boundary_proj.plot(
            ax=ax,
            facecolor="#FFF1B8",
            edgecolor="none",
            alpha=0.62,
            zorder=0.7,
        )
    draw_gaia_builtup(ax, bounds, clip_boundary=boundary, project_to_albers=True)
    draw_flood_point_kde(ax, pts, bounds, clip_boundary=boundary, project_to_albers=True)

    if boundary_proj is not None and not boundary_proj.empty:
        boundary_proj.boundary.plot(ax=ax, color="#4F4F4F", linewidth=0.72, alpha=0.94, zorder=5)

    if not pts.empty:
        xmin, ymin, xmax, ymax = bounds
        plot_pts = pts[
            pts["lon"].between(xmin, xmax)
            & pts["lat"].between(ymin, ymax)
        ].copy()
        if len(plot_pts) > 1000:
            plot_pts = plot_pts.sample(n=1000, random_state=seed + 416)
        plot_x, plot_y = project_lonlat(plot_pts["lon"], plot_pts["lat"])
        ax.scatter(
            plot_x, plot_y,
            s=2.4 if len(plot_pts) < 80 else 1.55,
            color="#1f6f8b",
            edgecolors="none",
            linewidths=0,
            alpha=0.52,
            zorder=6,
        )

    ax.text(
        0.0, 1.018,
        f"({letter}) {title}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
        zorder=8,
    )

    ax.set_xlim(proj_bounds[0], proj_bounds[2])
    ax.set_ylim(proj_bounds[1], proj_bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_city_inset(ax: plt.Axes, title: str, letter: str, seed: int = 0) -> None:
    """
    Representative city-event inset.

    Placeholder:
        city boundary + built-up grids + flood report/event points

    Replace with real city-event grid maps:
        city boundary
        GAIA built-up footprint
        flood report points and event centre
        optional faint report-density cue
    """
    rng = np.random.default_rng(seed)

    style_panel(ax, facecolor="none", border=False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    boundary = np.array([
        [1.0, 1.4], [1.8, 0.8], [3.4, 0.7], [5.5, 1.1], [7.8, 1.0],
        [9.0, 2.3], [9.2, 4.6], [8.4, 7.2], [6.3, 9.1], [4.1, 9.3],
        [2.0, 8.5], [0.8, 6.5], [0.7, 3.7], [1.0, 1.4],
    ])

    # Built-up grid
    for i in range(10):
        for j in range(10):
            if rng.random() < 0.62:
                ax.add_patch(
                    Rectangle(
                        (i, j), 0.86, 0.86,
                        facecolor="#787878",
                        edgecolor="white",
                        linewidth=0.25,
                        alpha=0.42
                    )
                )

    ax.plot(boundary[:, 0], boundary[:, 1], color="#5f5f5f", linewidth=0.75, alpha=0.90)

    report_x = np.clip(rng.normal(5.4, 1.25, 38), 1.2, 8.6)
    report_y = np.clip(rng.normal(5.1, 1.15, 38), 1.1, 8.7)
    ax.hexbin(report_x, report_y, gridsize=10, extent=(0, 10, 0, 10), mincnt=1, cmap="Greys", alpha=0.12, linewidths=0)
    ax.scatter(report_x, report_y, s=4.5, color="#1f6f8b", edgecolors="white", linewidths=0.10, alpha=0.86, zorder=4)

    ax.text(
        0.0, 1.018, f"({letter}) {title}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=7.1,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )


# =========================================================
# Panel a: concept
# =========================================================
def draw_concept_panel(ax: plt.Axes) -> None:
    """
    Panel a: conceptual framing.

    Message:
    Extreme rainfall can open additional event-time hotspot space outside
    recurrent cores, making static hotspot inventories miss population,
    road and everyday-function exposure.
    """
    style_panel(ax, facecolor="white", border=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("a.", loc="left", pad=10)

    blocks = [
        (0.07, "Recurrent hotspot cores", "static inventory"),
        (0.39, "Extreme rainfall event", "event-time expansion"),
        (0.71, "Hidden exposure burden", "population, roads and POIs"),
    ]

    for bx, title, subtitle in blocks:
        ax.add_patch(
            Rectangle(
                (bx, 0.25), 0.22, 0.48,
                facecolor="#F7F7F7", edgecolor="#BBBBBB", linewidth=1.0
            )
        )

        gx = bx + 0.035
        gy = 0.38
        cell = 0.028

        for i in range(5):
            for j in range(5):
                ax.add_patch(
                    Rectangle(
                        (gx + i * cell, gy + j * cell),
                        cell * 0.82, cell * 0.82,
                        facecolor="#E7E7E7", edgecolor="white", linewidth=0.25
                    )
                )

        recurrent_cells = [(1, 2), (2, 2), (2, 3)]
        for i, j in recurrent_cells:
            ax.add_patch(
                Rectangle(
                    (gx + i * cell, gy + j * cell),
                    cell * 0.82, cell * 0.82,
                    facecolor=COLORS["recurrent"], edgecolor="white", linewidth=0.25
                )
            )

        if bx >= 0.39:
            new_cells = [(3, 2), (3, 1), (4, 2), (1, 1), (2, 0)]
            for i, j in new_cells:
                ax.add_patch(
                    Rectangle(
                        (gx + i * cell, gy + j * cell),
                        cell * 0.82, cell * 0.82,
                        facecolor=COLORS["new"], edgecolor="white", linewidth=0.25
                    )
                )

        if bx >= 0.71:
            for i, j in [(4, 2), (3, 1), (2, 0)]:
                ax.add_patch(
                    Circle(
                        (gx + i * cell + cell * 0.40, gy + j * cell + cell * 0.40),
                        radius=0.010,
                        facecolor=COLORS["function"],
                        edgecolor="white",
                        linewidth=0.2
                    )
                )

        ax.text(
            bx + 0.11, 0.78, title,
            ha="center", va="center",
            fontsize=8.8,
            fontweight="bold"
        )
        ax.text(
            bx + 0.11, 0.17, subtitle,
            ha="center", va="center",
            fontsize=7.5,
            color=COLORS["gray_text"]
        )

    for x0, x1 in [(0.30, 0.38), (0.62, 0.70)]:
        ax.add_patch(
            FancyArrowPatch(
                (x0, 0.49), (x1, 0.49),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color="#333333"
            )
        )

    ax.text(
        0.50, 0.055,
        "Extreme rainfall opens additional hotspot space beyond historically recurrent cores.",
        ha="center", va="center",
        fontsize=7.5,
        color=COLORS["gray_text"],
    )


# =========================================================
# Panel b: workflow
# =========================================================
def draw_workflow_panel(ax: plt.Axes) -> None:
    """
    Panel b: data-processing and analytical workflow.
    """
    style_panel(ax, facecolor="white", border=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("b.", loc="left", pad=10)

    steps = [
        ("Weibo archive", "73.52M raw posts"),
        ("Qwen2.5\nextraction", "flood reports + locations"),
        ("Geocoding + QC", "151,172 flood points"),
        ("Rainfall\nlinkage", "462 extreme city-events"),
        ("Grid-event\nsystem", "864,131 observations"),
    ]

    xs = np.linspace(0.045, 0.805, len(steps))
    box_w = 0.140
    box_h = 0.34

    for i, ((title, subtitle), x) in enumerate(zip(steps, xs)):
        ax.add_patch(
            Rectangle(
                (x, 0.36), box_w, box_h,
                facecolor=COLORS["pipeline"],
                edgecolor=COLORS["pipeline_edge"],
                linewidth=1.0,
            )
        )
        ax.text(
            x + box_w / 2, 0.57, title,
            ha="center", va="center",
            fontsize=7.3,
            fontweight="bold",
            linespacing=0.88,
        )
        ax.text(
            x + box_w / 2, 0.43, subtitle,
            ha="center", va="center",
            fontsize=6.2,
            color=COLORS["gray_text"],
        )

        if i < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + box_w + 0.008, 0.53),
                    (xs[i + 1] - 0.012, 0.53),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=1.0,
                    color="#555555",
                )
            )

    badges = [
        ("253", "analytical cities"),
        ("180", "extreme-event cities"),
        ("3,397", "hotspot records"),
    ]
    for i, (value, label) in enumerate(badges):
        x = 0.18 + i * 0.28
        ax.text(
            x, 0.16,
            value,
            ha="center", va="center",
            fontsize=10.2,
            fontweight="bold",
            color=COLORS["pipeline_edge"],
        )
        ax.text(
            x, 0.075,
            label,
            ha="center", va="center",
            fontsize=6.7,
            color=COLORS["gray_text"],
        )


def draw_empty_panel(ax: plt.Axes, letter: str) -> None:
    """Reserve space for a manuscript panel that will be designed later."""
    style_panel(ax, facecolor="white", border=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(letter, loc="left", pad=10)


def draw_png_panel(ax: plt.Axes, png_path: str | Path, letter: str) -> None:
    """Draw a finished bitmap panel without stretching its aspect ratio."""
    style_panel(ax, facecolor="white", border=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    png_path = Path(png_path)
    if not png_path.exists():
        draw_empty_panel(ax, letter)
        ax.text(
            0.5,
            0.5,
            f"Missing image:\n{png_path}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.0,
            color=COLORS["gray_text"],
        )
        return

    img = mpimg.imread(png_path)
    # The supplied schematic PNGs have a very light off-white canvas; normalize
    # only near-white, low-saturation pixels so the two panels merge cleanly.
    if img.ndim == 3:
        rgb = img[..., :3]
        white_bg = (rgb.min(axis=2) > 0.955) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 0.018)
        img = img.copy()
        img[..., :3][white_bg] = 1.0
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()
    ax.text(
        0.012,
        0.985,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        color=COLORS["text"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.35),
        zorder=10,
    )


# =========================================================
# Main plotting function
# =========================================================
def plot_fig1_v5(
    output_dir: str | Path = REPO_ROOT / "outputs" / "figures",
    output_stem: str = "fig1_national_map_clean_png",
    full_table: str | Path = DEFAULT_FULL_TABLE,
    shp_path: str | Path = DEFAULT_SHP,
    point_table: str | Path = DEFAULT_POINT_TABLE,
    panel_a_png: str | Path = DEFAULT_PANEL_A_PNG,
    panel_b_png: str | Path = DEFAULT_PANEL_B_PNG,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress("loading repaired grid table")
    full_df = load_repaired_grid_table(full_table)
    progress("loading flood-report points")
    point_df = load_flood_points(point_table)
    progress("preparing national sample and administrative layers")
    national_sample = prepare_national_sample(full_df, shp_path)
    progress("loading South China Sea line layers")
    nine_dash = load_nine_dash_line()
    south_china_islands = load_south_china_islands()

    fig_w, fig_h = 11.0, 11.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # ---------------------------------------------------------
    # Manual boundaries
    # ---------------------------------------------------------
    L = 0.010
    R = 0.996
    T = 0.950
    B = 0.025

    # ---------------------------------------------------------
    # Top row: a and b
    # ---------------------------------------------------------
    top_y = 0.685
    top_h = T - top_y
    gap_top = 0.000

    a_x = L
    a_w = (R - L - gap_top) / 2

    b_x = a_x + a_w + gap_top
    b_w = a_w

    # ---------------------------------------------------------
    # Bottom region: c
    # ---------------------------------------------------------
    c_x = L
    c_y = B
    c_w = R - L
    c_h = top_y - c_y - 0.004

    # c container
    ax_cbox = fig.add_axes([c_x, c_y, c_w, c_h])
    style_panel(ax_cbox, facecolor="white", border=True)
    ax_cbox.spines["left"].set_visible(False)
    ax_cbox.spines["bottom"].set_visible(False)

    # Inner margins inside c
    inner_margin_x = 0.0005
    inner_margin_top = 0.000
    inner_margin_bottom = 0.028

    cx0 = c_x + inner_margin_x
    cx1 = c_x + c_w - inner_margin_x
    cy0 = c_y + inner_margin_bottom
    cy1 = c_y + c_h - inner_margin_top
    fig.text(
        c_x,
        cy1,
        "c",
        ha="left",
        va="top",
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        color=COLORS["text"],
    )

    # Layout inside c
    inset_w = 0.164
    inset_h = inset_w * fig_w / fig_h
    left_col_w = inset_w
    right_col_w = inset_w
    center_gap = 0.004
    bottom_row_h = inset_h
    mid_gap = 0.022

    center_x0 = cx0 + left_col_w + center_gap
    center_x1 = cx1 - right_col_w - center_gap
    center_w = center_x1 - center_x0
    map_y = cy0 + bottom_row_h + mid_gap
    map_h = cy1 - map_y
    pxmin, pymin, pxmax, pymax = project_bounds(MAIN_CHINA_BOUNDS)
    china_ratio = (pxmax - pxmin) / max(pymax - pymin, 1)
    map_w = min(center_w, map_h * fig_h / fig_w * china_ratio)
    map_x = center_x0 + (center_w - map_w) / 2

    left_w = left_col_w
    right_w = right_col_w
    side_map_gap = 0.012
    left_x = max(cx0, map_x - left_w - side_map_gap)
    right_x = min(cx1 - right_w, map_x + map_w + side_map_gap)

    side_gap_y = max(0.018, map_h - 2 * inset_h)
    side_inset_h = inset_h
    side_lower_shift = 0.038

    bottom_x = cx0
    bottom_w = cx1 - cx0
    bottom_y = cy0
    bottom_h = bottom_row_h

    # ---------------------------------------------------------
    # a
    # ---------------------------------------------------------
    progress("drawing panel a from PNG")
    ax_a = fig.add_axes([a_x, top_y, a_w, top_h])
    draw_png_panel(ax_a, panel_a_png, "a")

    # ---------------------------------------------------------
    # b
    # ---------------------------------------------------------
    progress("drawing panel b from PNG")
    ax_b = fig.add_axes([b_x, top_y, b_w, top_h])
    draw_png_panel(ax_b, panel_b_png, "b")

    # ---------------------------------------------------------
    # c center map
    # ---------------------------------------------------------
    progress("drawing national map")
    ax_map = fig.add_axes([map_x, map_y, map_w, map_h])
    style_panel(ax_map, facecolor=COLORS["map_bg"], border=True)
    draw_real_china_map(ax_map, national_sample, nine_dash=nine_dash, islands=south_china_islands)

    legend_handles = [
        Line2D(
            [0], [0],
            color="#1E1E1E",
            linewidth=1.3,
            label="National border"
        ),
        Line2D(
            [0], [0],
            color="#666666",
            linewidth=1.0,
            label="Province boundaries"
        ),
        Line2D(
            [0], [0],
            color="#A7A7A7",
            linewidth=0.9,
            label="City boundaries"
        ),
        Line2D(
            [0], [0],
            color="#C7771E",
            linewidth=1.8,
            label="Inset city boundaries"
        ),
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor="#D4D4D4", markeredgecolor="white", markeredgewidth=0.3,
            markersize=4.2,
            label="Analytical cities"
        ),
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor="#5B86B0", markeredgecolor="white", markeredgewidth=0.35,
            markersize=4.8,
            label="Extreme-event cities"
        ),
    ]
    ax_map.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
        fontsize=6.6,
        handlelength=1.6,
        borderpad=0.24,
        labelspacing=0.30,
    )

    # ---------------------------------------------------------
    # c insets around map
    # ---------------------------------------------------------
    # Left 2
    left_titles = [CITY_INSETS[0], CITY_INSETS[1]]
    top_inset_down_shift = 0.034
    for i, (title, city_name, letter) in enumerate(left_titles):
        progress(f"drawing inset {letter} {title}")
        y = map_y + map_h - (i + 1) * side_inset_h - i * side_gap_y
        if letter == "c1":
            y -= top_inset_down_shift
        if letter == "c2":
            y += side_lower_shift
        ax = fig.add_axes([left_x, y, left_w, side_inset_h])
        draw_real_city_inset(
            ax, title, city_name, letter,
            point_df=point_df,
            boundary=get_city_boundary(national_sample, city_name),
            seed=i + 2,
        )

    # Right 2
    right_titles = [CITY_INSETS[2], CITY_INSETS[3]]
    for i, (title, city_name, letter) in enumerate(right_titles):
        progress(f"drawing inset {letter} {title}")
        y = map_y + map_h - (i + 1) * side_inset_h - i * side_gap_y
        if letter == "c3":
            y -= top_inset_down_shift
        if letter == "c4":
            y += side_lower_shift
        ax = fig.add_axes([right_x, y, right_w, side_inset_h])
        draw_real_city_inset(
            ax, title, city_name, letter,
            point_df=point_df,
            boundary=get_city_boundary(national_sample, city_name),
            seed=i + 12,
        )

    # Bottom 6
    bottom_titles = CITY_INSETS[4:]
    bi_w = inset_w
    bi_gap = (bottom_w - 6 * bi_w) / 5

    for i, (title, city_name, letter) in enumerate(bottom_titles):
        progress(f"drawing inset {letter} {title}")
        x = bottom_x + i * (bi_w + bi_gap)
        ax = fig.add_axes([x, bottom_y, bi_w, inset_h])
        draw_real_city_inset(
            ax, title, city_name, letter,
            point_df=point_df,
            boundary=get_city_boundary(national_sample, city_name),
            seed=i + 20,
        )

    inset_legend = [
        Line2D([0], [0], color="#5f5f5f", linewidth=0.8, label="City boundary"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#999999", markeredgecolor="none", alpha=0.70, markersize=6, label="GAIA built-up"),
        Rectangle((0, 0), 1, 1, facecolor="#F4A08A", edgecolor="none", alpha=0.34, label="Flood-report density"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f6f8b", markeredgecolor="none", alpha=0.52, markersize=3.5, label="Flood reports"),
    ]
    fig.legend(
        handles=inset_legend,
        loc="lower right",
        bbox_to_anchor=(R - 0.004, bottom_y + 0.006),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
        fontsize=6.6,
        columnspacing=0.85,
        handletextpad=0.35,
        borderpad=0.25,
    )

    # ---------------------------------------------------------
    # Optional alignment guides
    # ---------------------------------------------------------
    SHOW_GUIDES = False
    if SHOW_GUIDES:
        guide_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
        guide_ax.axis("off")
        for x in [L, R, a_x, b_x, c_x, cx0, map_x, right_x, cx1]:
            guide_ax.plot([x, x], [0, 1], color="red", linewidth=0.5, alpha=0.35)
        for y in [T, top_y, c_y, cy0, map_y, cy1, B]:
            guide_ax.plot([0, 1], [y, y], color="red", linewidth=0.5, alpha=0.35)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    png_path = output_dir / f"{output_stem}.png"

    progress("saving PNG")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    progress(f"saved PNG: {png_path}")


if __name__ == "__main__":
    plot_fig1_v5(output_dir=REPO_ROOT / "outputs" / "figures", output_stem="Figure1")
