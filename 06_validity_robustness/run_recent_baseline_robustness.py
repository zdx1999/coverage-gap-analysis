#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hotspot_robustness_utils import (
    build_recent_window_outcome,
    prepare_full_footprint_sample,
    get_model_specs,
    run_event_fe_lpm,
    summarize_positive_rate,
    write_text_report,
)


def main():
    ap = argparse.ArgumentParser(description="Reclassify newly opened hotspots using recent non-extreme baselines and rerun full-footprint models.")
    ap.add_argument("--full-csv", required=True, help="Full city-event-grid table")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--lookbacks", default="3,5", help="Comma-separated recent baseline windows in years")
    ap.add_argument("--cluster", default="event", choices=["event", "city"], help="Cluster-robust SE level")
    ap.add_argument("--save-reclassified-full", action="store_true", help="Also save the full reclassified table (can be large)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(args.full_csv, low_memory=False)
    lookbacks = [int(x.strip()) for x in str(args.lookbacks).split(",") if x.strip()]

    all_results = []
    summaries = []

    base_sample, basic, covs = prepare_full_footprint_sample(full_df, "new_hotspot_region", cutoff_year=None)
    base_summary = summarize_positive_rate(base_sample, "new_hotspot_region", basic["event_id"], basic["city"])
    base_summary["sample_name"] = "original_history"
    summaries.append(base_summary)

    for model_name, preds in get_model_specs(base_sample).items():
        if not preds:
            continue
        res = run_event_fe_lpm(
            base_sample,
            outcome_col="new_hotspot_region",
            predictors=preds,
            event_col=basic["event_id"],
            city_col=basic["city"],
            cluster_col=args.cluster,
            model_name=model_name,
        )
        res["sample_name"] = "original_history"
        all_results.append(res)

    for lb in lookbacks:
        alt_col = f"new_hotspot_recent_{lb}y"
        rec = build_recent_window_outcome(full_df, lookback_years=lb, output_col=alt_col)

        if args.save_reclassified_full:
            rec_path = out_dir / f"reclassified_full_{lb}y.csv"
            rec.to_csv(rec_path, index=False, encoding="utf-8-sig")
            print(f"Saved: {rec_path}")

        sample, basic, covs = prepare_full_footprint_sample(rec, alt_col, cutoff_year=None)
        summary = summarize_positive_rate(sample, alt_col, basic["event_id"], basic["city"])
        summary["sample_name"] = f"recent_{lb}y"
        summaries.append(summary)

        for model_name, preds in get_model_specs(sample).items():
            if not preds:
                continue
            res = run_event_fe_lpm(
                sample,
                outcome_col=alt_col,
                predictors=preds,
                event_col=basic["event_id"],
                city_col=basic["city"],
                cluster_col=args.cluster,
                model_name=model_name,
            )
            res["sample_name"] = f"recent_{lb}y"
            all_results.append(res)

    coef_df = pd.concat(all_results, ignore_index=True)
    coef_csv = out_dir / "recent_baseline_fullfootprint_coefficients.csv"
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
    compact_csv = out_dir / "recent_baseline_fullfootprint_coefficients_for_plot.csv"
    compact.to_csv(compact_csv, index=False, encoding="utf-8-sig")

    write_text_report(
        str(out_dir / "recent_baseline_sample_summary.txt"),
        "Recent-baseline robustness summary",
        summaries,
        note="Compare original-history newly opened hotspot definition with recent-window baseline definitions (for example 3y and 5y). Focus on whether the full-footprint gradients remain directionally stable.",
    )

    print(f"Saved: {coef_csv}")
    print(f"Saved: {compact_csv}")
    print(f"Saved: {out_dir / 'recent_baseline_sample_summary.txt'}")


if __name__ == "__main__":
    main()
