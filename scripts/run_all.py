#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the main coverage-gap workflow and regenerate manuscript figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


STEPS = [
    ["scripts/coverage_gap_analysis/run_coverage_gap_analysis.py"],
    ["scripts/coverage_gap_analysis/run_coverage_gap_validity_checks.py"],
    ["scripts/figures/fig1_national_framework.py"],
    ["scripts/figures/fig2_coverage_gap.py"],
    ["scripts/figures/fig3_mechanisms.py"],
    ["scripts/figures/fig4_exposure.py"],
    ["scripts/figures/fig5_priorities.py"],
]


def main() -> None:
    for step in STEPS:
        cmd = [sys.executable, *step]
        print("\n[run_all]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
