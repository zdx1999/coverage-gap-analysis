# Historical flood hotspot inventory coverage gap

This repository contains the analysis and figure-generation code for the manuscript:

**Historical flood hotspot inventories leave widespread coverage gaps under extreme rainfall**

The study evaluates whether historical urban flood hotspot inventories cover reported flood-impact footprints during extreme city-rainfall events in China. The core analysis uses a leave-one-event-out historical inventory, where the focal event is excluded before ranking historical-report grids within the same city.

## Main result reproduced by this code

- 462 extreme city-events across 177 cities.
- Historical inventory: top 20% positive historical-report grids within the same city, excluding the focal event.
- Event footprint: 1 km built-up grids with `flood_count > 0`.
- Median uncovered impact share: 77.0%.
- Events with uncovered impact share >25%: 98.1%.
- Events with uncovered impact share >50%: 88.5%.
- Median exposure outside historical hotspots: population 73.7%, roads 73.5%, urban functions 71.4%.
- Correlation between uncovered impact share and function exposure outside historical hotspots: Pearson/Spearman 0.86.

## Repository structure

```text
scripts/
  upstream_workflow/                      # Weibo, LLM, geocoding, GEE and event-building workflow
    01_weibo_preprocessing/
    02_gee_extraction/
    03_event_construction/
    05_temporal_robustness/
  coverage_gap_analysis/
    run_coverage_gap_analysis.py          # main coverage-gap analysis
    run_coverage_gap_validity_checks.py   # random baseline and robustness checks
  figures/
    fig1_national_framework.py
    fig2_coverage_gap.py
    fig3_mechanisms.py
    fig4_exposure.py
    fig5_priorities.py
  support/
    fig2_zhengzhou_5551_final_refined_v5.py
    fig2a_national_footprint.py
  run_all.py
data/
  README.md                               # expected input data schema and filenames
outputs/
  figures/                                # final manuscript figures as PNG previews
  tables/                                 # selected derived result tables
docs/
  code_availability_template.md
  figure_overview.md
  technical_note_validity_checks.md
```

## Data availability

Large raw social-media, geospatial and gridded exposure files are not included in this code repository. The code expects processed analysis-ready files in `data/`; see `data/README.md` for required filenames and column schemas.

The upstream workflow scripts are provided under `scripts/upstream_workflow/` to document the processing chain from Weibo record screening, LLM-based information extraction and geocoding to Google Earth Engine covariate extraction and event-panel construction. The public repository does not include raw Weibo text, user identifiers, API keys, or restricted third-party geospatial datasets.

If the public release uses a data repository such as Zenodo, Figshare or OSF, place the DOI and download instructions here before submission.

## Environment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Some map panels use optional geospatial packages and local shapefiles. If `contextily`, `geopandas`, `cartopy` or local shapefiles are unavailable, the analysis tables can still be reproduced, but map styling may be simplified.

## Reproduction workflow

After placing the required input files under `data/`, run:

```bash
python scripts/run_all.py
```

Or run the steps manually:

```bash
python scripts/coverage_gap_analysis/run_coverage_gap_analysis.py
python scripts/coverage_gap_analysis/run_coverage_gap_validity_checks.py
python scripts/figures/fig1_national_framework.py
python scripts/figures/fig2_coverage_gap.py
python scripts/figures/fig3_mechanisms.py
python scripts/figures/fig4_exposure.py
python scripts/figures/fig5_priorities.py
```

The main analysis writes derived tables to `outputs/coverage_gap_analysis/`. Figure scripts write PNG/PDF outputs and figure-specific tables to `outputs/figures/`.

## Interpretation boundary

The reported flood-impact footprint is derived from public social-media reports. It should be interpreted as a socially sensed impact footprint rather than a complete hydrodynamic inundation boundary. Exposure metrics quantify population, road and urban-function exposure located outside historical hotspot inventories; they are not direct loss estimates.

## Citation

Please cite the manuscript and data repository when available.


## Upstream social-media and LLM workflow

The upstream workflow is included for transparency and reproducibility of the data-processing logic:

```text
scripts/upstream_workflow/
  01_weibo_preprocessing/       # Weibo screening, LLM extraction, location cleaning and geocoding notebook
  02_gee_extraction/            # GEE rainfall, night-time lights, population and terrain extraction
  03_event_construction/        # event metadata, extreme-event definition and heterogeneity panels
  05_temporal_robustness/       # recent-baseline and late-period robustness checks
  docs/file_manifest.csv        # source-to-package manifest from the original submission code package
```

The notebooks are distributed with execution outputs cleared. Configure local paths, vLLM/OpenAI-compatible endpoints,
Amap/GEE credentials and data-access permissions before running the upstream workflow.
