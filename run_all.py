#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the main coverage-gap workflow and regenerate manuscript figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


STEPS = [
    ["04_coverage_gap_analysis/run_coverage_gap_analysis.py"],
    ["06_validity_robustness/run_coverage_gap_validity_checks.py"],
    ["05_figure_generation/fig1_national_framework.py"],
    ["05_figure_generation/fig2_coverage_gap.py"],
    ["05_figure_generation/fig3_mechanisms.py"],
    ["05_figure_generation/fig4_exposure.py"],
    ["05_figure_generation/fig5_priorities.py"],
]


def main() -> None:
    for step in STEPS:
        cmd = [sys.executable, *step]
        print("\n[run_all]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
