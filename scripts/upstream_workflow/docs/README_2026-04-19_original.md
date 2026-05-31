# Submission Code Package (2026-04-19, updated 2026-04-22)

This folder is a curated code package prepared for manuscript submission.
It is organized around the end-to-end workflow used in the paper:
1) Weibo preprocessing
2) GEE extraction
3) Event construction and heterogeneity checks
4) Final figure generation
5) Time-distribution robustness checks

## Structure
- `01_weibo_preprocessing/`
  - `全国城市处理.ipynb`
- `02_gee_extraction/`
  - `降水提取.ipynb`
  - `extract_ntl_pop_with_gee_api.py`
  - `extract_copdem_terrain_twi_with_gee_api.py`
  - `merge_gee_back_to_city_grid_base_gaia.py`
- `03_event_construction/`
  - `prepare_event_meta_with_builtup_rain.py`
  - `define_extreme_and_new_hotspots.py`
  - `run_city_heterogeneity_reframed_v2.py`
- `04_figure_generation/`
  - final plotting scripts (Figure 1c, 2, 3, 4, 5)
- `05_temporal_robustness/`
  - late-period / recent-baseline robustness scripts for uneven Weibo time distribution
- `docs/`
  - `file_manifest.csv` (source-to-package mapping)
  - `CODE_AVAILABILITY_DRAFT.md` (manuscript-ready text draft)

## Notes
- This package copies scripts/notebooks only; original files remain unchanged.
- Some scripts contain project-specific local paths and external data dependencies.
- Data files are not duplicated here to avoid very large package size.
- Current recommended figure scripts in this package include:
  - `figure3a_forest_plot_v3_legend_topright.py`
  - `figure4_hidden_exposure_burden_v2.py`
  - `main_fig5_functional_disruption_heterogeneity_final.py`

## Minimal run order (recommended)
1. Run Weibo preprocessing notebook (`01_weibo_preprocessing`).
2. Run rainfall + GEE extraction scripts (`02_gee_extraction`).
3. Build event-level tables and extreme-event/new-hotspot definitions (`03_event_construction`).
4. Generate publication figures (`04_figure_generation`).
5. Run time-distribution robustness checks (`05_temporal_robustness`) if needed for SI.
