#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hotspot_robustness_utils import (
    prepare_full_footprint_sample,
    get_model_specs,
    run_event_fe_lpm,
    summarize_positive_rate,
    write_text_report,
)


def main():
    ap = argparse.ArgumentParser(description="Run late-period-only full-footprint robustness for newly opened hotspot formation.")
    ap.add_argument("--full-csv", required=True, help="Full city-event-grid table")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--outcome-col", default="new_hotspot_region", help="Outcome column to model")
    ap.add_argument("--cutoffs", default="2018,2019", help="Comma-separated cutoff years, e.g. 2018,2019")
    ap.add_argument("--cluster", default="event", choices=["event", "city"], help="Cluster-robust SE level")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(args.full_csv, low_memory=False)
    cutoffs = [int(x.strip()) for x in str(args.cutoffs).split(",") if x.strip()]

    all_results = []
    summaries = []

    for cutoff in [None] + cutoffs:
        sample, basic, covs = prepare_full_footprint_sample(full_df, args.outcome_col, cutoff_year=cutoff)
        specs = get_model_specs(sample)
        label = "all_years" if cutoff is None else f"post_{cutoff}"
        summary = summarize_positive_rate(sample, args.outcome_col, basic["event_id"], basic["city"])
        summary["sample_name"] = label
        summaries.append(summary)

        for model_name, preds in specs.items():
            if not preds:
                continue
            res = run_event_fe_lpm(
                sample,
                outcome_col=args.outcome_col,
                predictors=preds,
                event_col=basic["event_id"],
                city_col=basic["city"],
                cluster_col=args.cluster,
                model_name=model_name,
            )
            res["sample_name"] = label
            all_results.append(res)

    coef_df = pd.concat(all_results, ignore_index=True)
    coef_csv = out_dir / "late_period_fullfootprint_coefficients.csv"
    coef_df.to_csv(coef_csv, index=False, encoding="utf-8-sig")

    focus_terms = {
        "z_distance_to_centre": "distance_to_centre",
        "z_night_time_lights": "night_time_lights",
        "z_population": "population",
        "z_poi_log1p": "poi_log1p",
        "z_poi_category_richness": "poi_category_richness",
    }
    compact = coef_df[coef_df["term"].isin(focus_terms)].copy()
    compact["term"] = compact["term"].map(focus_terms)
    compact_csv = out_dir / "late_period_fullfootprint_coefficients_for_plot.csv"
    compact.to_csv(compact_csv, index=False, encoding="utf-8-sig")

    write_text_report(
        str(out_dir / "late_period_sample_summary.txt"),
        "Late-period robustness summary",
        summaries,
        note="Compare sign stability across all years and post-cutoff samples. Focus on whether distance_to_centre remains negative and night_time_lights / population / POI terms remain positive.",
    )

    print(f"Saved: {coef_csv}")
    print(f"Saved: {compact_csv}")
    print(f"Saved: {out_dir / 'late_period_sample_summary.txt'}")


if __name__ == "__main__":
    main()
