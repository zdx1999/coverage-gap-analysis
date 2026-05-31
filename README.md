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
  five_figures_structure_content_plan.md
  technical_note_validity_checks.md
```

## Data availability

Large raw social-media, geospatial and gridded exposure files are not included in this code repository. The code expects processed analysis-ready files in `data/`; see `data/README.md` for required filenames and column schemas.

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

