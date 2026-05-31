# Code Availability Text

## Main Manuscript

Code used to preprocess social-media flood records, run LLM-assisted information extraction and geocoding, extract geospatial covariates with Google Earth Engine, construct event-level analytical panels, reproduce the coverage-gap analysis, and generate all main figures is available at https://github.com/zdx1999/coverage-gap-analysis. The repository excludes raw Weibo text, user identifiers, API keys and restricted third-party geospatial data; processed, de-identified analysis-ready tables should be supplied through the associated data repository when available.

## Supplementary Information

The public code repository is organized as a numbered workflow: Weibo preprocessing (`01_weibo_preprocessing/`), GEE covariate extraction (`02_gee_extraction/`), event construction (`03_event_construction/`), coverage-gap analysis (`04_coverage_gap_analysis/`), figure generation (`05_figure_generation/`), and validity/robustness checks (`06_validity_robustness/`). The upstream workflow documents Weibo screening, LLM-based extraction of flood status, time and place information, geocoding, GEE extraction of rainfall, night-time lights, population and terrain variables, event construction, and temporal robustness checks. The final workflow reproduces the leave-one-event-out historical inventory, coverage-gap statistics, validity checks and Figs. 1-5 from processed analysis-ready tables.
