# Code Availability Draft (for manuscript)

## Main Manuscript (short version)
The code used for Weibo-based event preprocessing, Google Earth Engine (GEE) covariate extraction, event construction, and figure generation is archived in a curated submission package and will be made publicly available upon publication (GitHub + Zenodo DOI to be inserted).

## Main Manuscript (Nature-style concise variant)
Code used to preprocess social-media flood records, extract geospatial covariates with Google Earth Engine, construct event-level analytical panels, and reproduce all main figures is available at [GitHub repository URL] and archived at Zenodo ([DOI], to be provided upon acceptance).

## Supplementary Information (expanded version)
We provide a structured code workflow covering: (i) national Weibo data preprocessing (`全国城市处理.ipynb`), (ii) GEE-based extraction of rainfall, night-time lights, population, and terrain-related variables (`降水提取.ipynb`, `extract_ntl_pop_with_gee_api.py`, `extract_copdem_terrain_twi_with_gee_api.py`, `merge_gee_back_to_city_grid_base_gaia.py`), (iii) event-level peak-rain integration and extreme-event/new-hotspot definition (`prepare_event_meta_with_builtup_rain.py`, `define_extreme_and_new_hotspots.py`), (iv) figure-generation scripts for the final manuscript figures (Figure 1c–5 scripts under `04_figure_generation/`), and (v) time-distribution robustness scripts for uneven Weibo posting intensity (`05_temporal_robustness/`).

A full file manifest is provided in `docs/file_manifest.csv`.

## Chinese working draft (for internal editing)
本文所用代码覆盖微博事件清洗、GEE变量提取、事件级样本构建以及最终出图流程。投稿版本代码已整理为结构化代码包，并将在论文接收后公开（建议同时提供 GitHub 仓库与 Zenodo DOI）。

可在正文中使用短版：
“用于微博数据处理、GEE变量提取、事件样本构建和图件复现的代码，将在论文发表时通过 GitHub 和 Zenodo 公开。”

可在 SI 中使用长版：
“代码包包含：全国微博预处理（`全国城市处理.ipynb`）；GEE提取流程（`降水提取.ipynb`、`extract_ntl_pop_with_gee_api.py`、`extract_copdem_terrain_twi_with_gee_api.py`、`merge_gee_back_to_city_grid_base_gaia.py`）；事件构建与极端事件/新热点定义（`prepare_event_meta_with_builtup_rain.py`、`define_extreme_and_new_hotspots.py`）；主文图件脚本（`04_figure_generation/`）；以及微博时间分布不均的稳健性检验脚本（`05_temporal_robustness/`）。完整清单见 `docs/file_manifest.csv`。”
