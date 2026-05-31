#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_LATE_CSV = Path(r"code416\outputs_time_robustness_test\late\late_period_fullfootprint_coefficients.csv")
DEFAULT_RECENT_CSV = Path(r"code416\outputs_time_robustness_test\recent\recent_baseline_fullfootprint_coefficients.csv")
DEFAULT_OUT_DIR = Path(r"code416\outputs_time_robustness_test\si_tables")

TERM_ORDER = [
    "distance_to_centre",
    "night_time_lights",
    "population",
    "poi_log1p",
    "poi_category_richness",
    "twi",
    "elevation",
    "slope",
    "development_year",
]

SAMPLE_ORDER = [
    "all_years",
    "post_2018",
    "post_2019",
    "original_history",
    "recent_3y",
    "recent_5y",
]


def normalize_term(term: str) -> str:
    t = str(term)
    if t.startswith("z_"):
        return t[2:]
    return t


def term_rank(term: str) -> int:
    t = normalize_term(term)
    if t in TERM_ORDER:
        return TERM_ORDER.index(t)
    return len(TERM_ORDER) + 100


def sample_rank(sample: str) -> int:
    s = str(sample)
    if s in SAMPLE_ORDER:
        return SAMPLE_ORDER.index(s)
    return len(SAMPLE_ORDER) + 100


def enrich(df: pd.DataFrame, family: str) -> pd.DataFrame:
    out = df.copy()
    out["robustness_family"] = family
    out["term_clean"] = out["term"].map(normalize_term)
    out["t_stat"] = out["coef"] / out["se"]
    out["abs_t"] = out["t_stat"].abs()
    out["sign"] = out["coef"].apply(lambda x: "+" if x > 0 else ("-" if x < 0 else "0"))
    out["p_lt_0_10"] = out["abs_t"] >= 1.645
    out["p_lt_0_05"] = out["abs_t"] >= 1.960
    out["p_lt_0_01"] = out["abs_t"] >= 2.576
    out["sig_star"] = out["abs_t"].apply(lambda x: "***" if x >= 2.576 else ("**" if x >= 1.960 else ("*" if x >= 1.645 else "")))
    out["sample_rank"] = out["sample_name"].map(sample_rank)
    out["term_rank"] = out["term_clean"].map(term_rank)
    return out


def build_sign_stability(df: pd.DataFrame) -> pd.DataFrame:
    core_terms = ["distance_to_centre", "night_time_lights", "population", "poi_log1p", "poi_category_richness"]
    sub = df[df["term_clean"].isin(core_terms)].copy()
    wide = (
        sub.pivot_table(
            index=["model", "term_clean"],
            columns="sample_name",
            values="sign",
            aggfunc="first",
        )
        .reset_index()
    )

    sample_cols = [c for c in SAMPLE_ORDER if c in wide.columns]

    def _all_same_sign(row: pd.Series) -> bool:
        vals: List[str] = [str(row[c]) for c in sample_cols if pd.notna(row[c]) and str(row[c]) in {"+", "-"}]
        if not vals:
            return False
        return len(set(vals)) == 1

    def _stable_sign(row: pd.Series) -> str:
        vals: List[str] = [str(row[c]) for c in sample_cols if pd.notna(row[c]) and str(row[c]) in {"+", "-"}]
        if not vals:
            return "NA"
        return vals[0] if len(set(vals)) == 1 else "mixed"

    wide["stable_across_samples"] = wide.apply(_all_same_sign, axis=1)
    wide["stable_sign"] = wide.apply(_stable_sign, axis=1)
    wide = wide.sort_values(["model", "term_clean"]).reset_index(drop=True)
    return wide


def build_sample_overview(df: pd.DataFrame) -> pd.DataFrame:
    overview = (
        df.groupby(["robustness_family", "sample_name", "model"], as_index=False)
        .agg(n_obs=("n", "first"), n_events=("events", "first"), n_cities=("cities", "first"))
    )
    overview["sample_rank"] = overview["sample_name"].map(sample_rank)
    overview = overview.sort_values(["sample_rank", "robustness_family", "model"]).drop(columns=["sample_rank"])
    return overview


def build_core_table(df: pd.DataFrame) -> pd.DataFrame:
    core_terms = ["distance_to_centre", "night_time_lights", "population", "poi_log1p", "poi_category_richness"]
    core = df[df["term_clean"].isin(core_terms)].copy()
    core = core.sort_values(["sample_rank", "model", "term_rank"])
    keep_cols = [
        "robustness_family",
        "sample_name",
        "model",
        "term_clean",
        "coef",
        "se",
        "t_stat",
        "sign",
        "sig_star",
        "n",
        "events",
        "cities",
    ]
    return core[keep_cols]


def write_text_note(path: Path, sample_overview: pd.DataFrame, sign_stability: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("Time-distribution robustness summary (for SI)")
    lines.append("============================================")
    lines.append("")
    lines.append("Sample sizes")
    for _, r in sample_overview.iterrows():
        lines.append(
            f"- {r['sample_name']} ({r['model']}): n={int(r['n_obs'])}, events={int(r['n_events'])}, cities={int(r['n_cities'])}"
        )
    lines.append("")

    lines.append("Sign stability of core gradients across samples")
    for model in sorted(sign_stability["model"].unique()):
        sub = sign_stability[sign_stability["model"] == model]
        stable_n = int(sub["stable_across_samples"].sum())
        total_n = int(len(sub))
        lines.append(f"- {model}: {stable_n}/{total_n} core terms keep a consistent sign across all available samples.")
        for _, r in sub.iterrows():
            lines.append(f"  - {r['term_clean']}: {r['stable_sign']}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare SI-ready summary tables for time-distribution robustness checks.")
    ap.add_argument("--late-csv", default=str(DEFAULT_LATE_CSV), help="Late-period robustness coefficient table")
    ap.add_argument("--recent-csv", default=str(DEFAULT_RECENT_CSV), help="Recent-baseline robustness coefficient table")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for SI summary tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    late = pd.read_csv(args.late_csv, low_memory=False)
    recent = pd.read_csv(args.recent_csv, low_memory=False)

    merged = pd.concat(
        [
            enrich(late, "late_period"),
            enrich(recent, "recent_baseline"),
        ],
        ignore_index=True,
    )
    merged = merged.sort_values(["sample_rank", "model", "term_rank"]).reset_index(drop=True)

    long_cols = [
        "robustness_family",
        "sample_name",
        "model",
        "term",
        "term_clean",
        "coef",
        "se",
        "t_stat",
        "sign",
        "sig_star",
        "p_lt_0_10",
        "p_lt_0_05",
        "p_lt_0_01",
        "n",
        "events",
        "cities",
    ]
    long_path = out_dir / "si_time_robustness_coefficients_long.csv"
    merged[long_cols].to_csv(long_path, index=False, encoding="utf-8-sig")

    sample_overview = build_sample_overview(merged)
    overview_path = out_dir / "si_time_robustness_sample_overview.csv"
    sample_overview.to_csv(overview_path, index=False, encoding="utf-8-sig")

    core = build_core_table(merged)
    core_path = out_dir / "si_time_robustness_core_terms.csv"
    core.to_csv(core_path, index=False, encoding="utf-8-sig")

    sign_stability = build_sign_stability(merged)
    sign_path = out_dir / "si_time_robustness_sign_stability.csv"
    sign_stability.to_csv(sign_path, index=False, encoding="utf-8-sig")

    note_path = out_dir / "si_time_robustness_summary.txt"
    write_text_note(note_path, sample_overview, sign_stability)

    print(f"Saved: {long_path}")
    print(f"Saved: {overview_path}")
    print(f"Saved: {core_path}")
    print(f"Saved: {sign_path}")
    print(f"Saved: {note_path}")


if __name__ == "__main__":
    main()

