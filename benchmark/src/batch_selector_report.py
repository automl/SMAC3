"""Summarizes the batch selector benchmark from its raw results.

Kept separate from the runner so the statistics can be reworked while a run is still going:

    python src/batch_selector_report.py [--stuck-threshold 0.01]
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

RAW_FILENAME = Path("report/batch_selector_raw.json")
BASELINE = "top_q"


def final_costs(runs: dict) -> dict[str, float]:
    return {seed: run["trajectory"][-1] for seed, run in runs.items() if run.get("trajectory")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stuck-threshold",
        type=float,
        default=0.01,
        help="a run whose final cost exceeds this counts as having failed to converge",
    )
    args = parser.parse_args()

    with open(RAW_FILENAME) as f:
        data = json.load(f)

    for task_name, arms in data.items():
        baseline = final_costs(arms.get(BASELINE, {}))
        complete = all(len(final_costs(runs)) == len(baseline) for runs in arms.values())
        status = "" if complete else "  (PARTIAL)"

        print()
        print(f"### {task_name}{status}")
        print(
            f"{'arm':14s} {'n':>3s} {'median':>10s} {'IQR':>20s} {'mean':>10s} "
            f"{'stuck':>7s} {'W-L-T':>9s} {'sign p':>8s} {'time':>7s}"
        )

        for arm, runs in arms.items():
            finals = final_costs(runs)
            if not finals:
                continue

            values = np.array(list(finals.values()))
            times = np.array([r["walltime"] for r in runs.values()])
            n_stuck = int((values > args.stuck_threshold).sum())

            if arm == BASELINE:
                record, pvalue = "-", ""
            else:
                shared = sorted(set(finals) & set(baseline))
                wins = sum(finals[s] < baseline[s] for s in shared)
                losses = sum(finals[s] > baseline[s] for s in shared)
                ties = len(shared) - wins - losses
                record = f"{wins}-{losses}-{ties}"
                # Two-sided sign test on the paired wins, ignoring ties.
                decided = wins + losses
                pvalue = (
                    f"{binomtest(wins, decided, 0.5).pvalue:.3f}" if decided else "n/a"
                )

            print(
                f"{arm:14s} {len(values):3d} {np.median(values):10.3g} "
                f"[{np.quantile(values, 0.25):8.3g},{np.quantile(values, 0.75):9.3g}] "
                f"{np.mean(values):10.3g} {n_stuck:4d}/{len(values):<2d} {record:>9s} "
                f"{pvalue:>8s} {np.mean(times):6.1f}s"
            )

    print()
    print(f"stuck = final cost > {args.stuck_threshold}; W-L-T is per-seed against {BASELINE}.")
    print("sign p is a two-sided sign test on the paired wins, which ignores effect size.")


if __name__ == "__main__":
    main()
