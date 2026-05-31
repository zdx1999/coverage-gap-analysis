# Technical note: coverage-gap validity checks

## 1. Random inventory baseline
- Median observed impact coverage by historical inventory is 23.0%, compared with random inventory 1.5%.
- Median coverage lift is 11.23x; 71.6% of events have empirical p < 0.05 for historical coverage exceeding random coverage.
- Interpretation: historical inventories are informative relative to random inventories, but their absolute coverage remains low.

## 2. Low-sample event robustness
- Across all low-sample filters, the minimum median uncovered impact share is 75.0%; the minimum median function exposure outside is 65.7%.
- all_events: n=462, median gap=77.0%, median function outside=71.4%, Pearson/Spearman=0.86/0.86.
- total_reports_ge_5: n=439, median gap=76.4%, median function outside=69.9%, Pearson/Spearman=0.85/0.85.
- total_reports_ge_10: n=401, median gap=75.9%, median function outside=68.9%, Pearson/Spearman=0.84/0.84.
- total_reports_ge_20: n=277, median gap=75.0%, median function outside=65.7%, Pearson/Spearman=0.86/0.84.
- affected_grids_ge_3: n=441, median gap=76.4%, median function outside=69.9%, Pearson/Spearman=0.84/0.84.
- affected_grids_ge_5: n=407, median gap=76.4%, median function outside=69.3%, Pearson/Spearman=0.85/0.84.
- affected_grids_ge_10: n=304, median gap=75.0%, median function outside=67.6%, Pearson/Spearman=0.86/0.85.

## 3. Split-half historical inventory stability
- Among cities with enough history, median split-half Jaccard is 0.14; median overlap coefficient is 0.32.
- Low-history cities account for 41.5% of cities and should be flagged or down-weighted in narrative examples.
- Excluding low-history cities leaves 413 extreme events across 139 cities; median uncovered impact remains 75.0%, with median function exposure outside 66.9%.
- Low-history cities alone have median uncovered impact 100.0%; this means they are not the sole source of the national coverage-gap result.

## 4. Gi* representative-city validation
- Median Gi*-based uncovered impact share is 90.0%, versus top20 median 84.0% for the same representative events.
- Median Gi*/top20 overlap coefficient is 0.33.
- Zhengzhou Event 5551: Gi* hotspots=191, top20 hotspots=89, overlap coefficient=0.61, Gi* gap=77.4%, top20 gap=86.8%.
- Guangzhou Event 4520: Gi* hotspots=78, top20 hotspots=79, overlap coefficient=0.44, Gi* gap=90.0%, top20 gap=84.0%.
- Beijing Event 7098: Gi* hotspots=87, top20 hotspots=139, overlap coefficient=0.25, Gi* gap=96.5%, top20 gap=87.1%.
- Wuhan Event 3856: Gi* hotspots=72, top20 hotspots=75, overlap coefficient=0.33, Gi* gap=85.4%, top20 gap=76.7%.
- Chengdu Event 3215: Gi* hotspots=54, top20 hotspots=93, overlap coefficient=0.26, Gi* gap=94.5%, top20 gap=81.4%.

## 5. Extreme vs non-extreme comparison
- Extreme events: median uncovered impact 77.0%, median function outside 71.4%.
- Non-extreme events: median uncovered impact 75.0%, median function outside 68.7%.
- Extreme events show a higher median spatial coverage gap than non-extreme events.

## Final judgement
- Passed 7/7 defensive criteria; final judgement: **strong support**.
- Recommendation: formally shift the paper mainline to historical hotspot inventory coverage gap.

## Interpretation boundary
- All footprints are public-reporting / socially sensed flood-impact footprints, not observed inundation polygons.
- Inventories are leave-one-event-out rather than strictly pre-event chronological; a chronology-only check remains a useful next robustness step if reliable event dates are harmonized.