# Upstream Social-Media and Geospatial Processing Workflow

This directory documents the upstream workflow that converts public Weibo-derived flood records into the processed analytical tables used by the coverage-gap analysis.

## Contents

- `01_weibo_preprocessing/`: national Weibo event-processing notebook, including text screening, LLM-assisted extraction, post-processing, geocoding and quality checks.
- `02_gee_extraction/`: Google Earth Engine extraction of rainfall, night-time lights, population and terrain-related covariates, plus merge-back scripts.
- `03_event_construction/`: event metadata construction, peak-rain integration, extreme-event definition and city heterogeneity panels.
- `05_temporal_robustness/`: robustness scripts for uneven Weibo temporal coverage.
- `docs/file_manifest.csv`: source-to-package manifest from the original curated submission code package.

## Public-release boundary

Raw Weibo text, user identifiers, API keys, local credentials and restricted third-party geospatial datasets are not included. The notebook outputs have been cleared before release. Users should configure local paths and credentials before execution.

The current manuscript figures and coverage-gap results are reproduced by the updated scripts under `scripts/coverage_gap_analysis/` and `scripts/figures/`; those scripts supersede older figure-generation files from the April 2026 submission package.
