"""Compares batch selectors against plain top-q selection on the synthetic benchmark models.

Run from the `benchmark` directory:

    python src/batch_selector_benchmark.py             # full run
    python src/batch_selector_benchmark.py --quick     # one task, two seeds, for timing

Results are appended to `report/batch_selector_raw.json` so an interrupted run can be resumed.
"""
from __future__ import annotations

# We don't want to create a "real" package here so we just work with this hack
import sys

sys.path.insert(0, ".")

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

logging.disable(9999)

from src.models.branin import Branin  # noqa: E402
from src.models.himmelblau import HimmelblauModel  # noqa: E402

RAW_FILENAME = Path("report/batch_selector_raw.json")
SEEDS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]


def _no_selector():
    return None


def _soft_rank(alpha: float) -> Callable:
    def build(seed: int = 0):
        from smac.acquisition.batch_selector import StochasticBatchSelector

        return StochasticBatchSelector(mode="soft_rank", alpha=alpha, seed=seed)

    return build


# Each arm is a way of turning the maximized candidates into the next batch.
ARMS: dict[str, Callable] = {
    "top_q": lambda seed=0: None,
    "soft_rank_a2": _soft_rank(2.0),
    "soft_rank_a1": _soft_rank(1.0),
}


@dataclass
class BatchTask:
    """One benchmark problem, configured so that a batch is actually formed."""

    name: str
    model_factory: Callable
    facade: str  # "hpo" (random forest) or "bb" (Gaussian process)
    n_trials: int
    retrain_after: int = 8
    n_workers: int = 1


TASKS = [
    BatchTask("Branin (HPO/RF)", Branin, "hpo", n_trials=200),
    BatchTask("Himmelblau (HPO/RF)", HimmelblauModel, "hpo", n_trials=300),
    BatchTask("Branin (BB/GP)", Branin, "bb", n_trials=100),
    BatchTask("Himmelblau (BB/GP)", HimmelblauModel, "bb", n_trials=100),
]


def run_once(task: BatchTask, arm: str, seed: int) -> dict:
    """Runs one optimization and returns its best-so-far trajectory and timing."""
    from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Scenario

    model = task.model_factory()
    facade_object = HyperparameterOptimizationFacade if task.facade == "hpo" else BlackBoxFacade

    scenario = Scenario(
        model.configspace,
        n_trials=task.n_trials,
        deterministic=True,
        n_workers=task.n_workers,
        seed=seed,
        output_directory=Path("smac3_output_batch_selector"),
    )

    smac = facade_object(
        scenario,
        model.train,
        config_selector=facade_object.get_config_selector(scenario, retrain_after=task.retrain_after),
        batch_selector=ARMS[arm](seed=seed),
        logging_level=99999,
        overwrite=True,
    )

    start = time.time()
    smac.optimize()
    elapsed = time.time() - start

    costs = []
    for config in smac.runhistory.get_configs():
        try:
            cost = smac.runhistory.get_cost(config)
        except Exception:
            continue
        if cost is not None and np.isfinite(cost):
            costs.append(float(cost))

    best_so_far = np.minimum.accumulate(costs).tolist() if costs else []

    return {"trajectory": best_so_far, "walltime": elapsed, "n_evaluated": len(costs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="one task, two seeds")
    args = parser.parse_args()

    tasks = TASKS[:1] if args.quick else TASKS
    seeds = SEEDS[:2] if args.quick else SEEDS

    RAW_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if RAW_FILENAME.exists():
        with open(RAW_FILENAME) as f:
            data = json.load(f)

    total = len(tasks) * len(ARMS) * len(seeds)
    done = 0
    for task in tasks:
        data.setdefault(task.name, {})
        for arm in ARMS:
            data[task.name].setdefault(arm, {})
            for seed in seeds:
                done += 1
                if str(seed) in data[task.name][arm]:
                    print(f"[{done}/{total}] {task.name} / {arm} / {seed}: cached")
                    continue

                result = run_once(task, arm, seed)
                data[task.name][arm][str(seed)] = result
                print(
                    f"[{done}/{total}] {task.name} / {arm} / {seed}: "
                    f"best={result['trajectory'][-1]:.4g} "
                    f"n={result['n_evaluated']} {result['walltime']:.1f}s"
                )

                with open(RAW_FILENAME, "w") as f:
                    json.dump(data, f)

    summarize(data, tasks)


def summarize(data: dict, tasks: list[BatchTask]) -> None:
    """Prints the final incumbent per arm and the paired difference against top-q."""
    print()
    print("=" * 96)
    print(f"{'task':22s} {'arm':14s} {'final cost (mean +- std)':28s} {'median':>10s} "
          f"{'wins/ties':>10s} {'time':>7s}")
    print("=" * 96)

    for task in tasks:
        if task.name not in data:
            continue

        baseline = {
            seed: run["trajectory"][-1]
            for seed, run in data[task.name].get("top_q", {}).items()
            if run["trajectory"]
        }

        for arm in ARMS:
            runs = data[task.name].get(arm, {})
            finals = np.array([r["trajectory"][-1] for r in runs.values() if r["trajectory"]])
            times = np.array([r["walltime"] for r in runs.values()])
            if len(finals) == 0:
                continue

            if arm == "top_q":
                record = "-"
            else:
                shared = [s for s in runs if s in baseline and runs[s]["trajectory"]]
                wins = sum(runs[s]["trajectory"][-1] < baseline[s] for s in shared)
                ties = sum(runs[s]["trajectory"][-1] == baseline[s] for s in shared)
                record = f"{wins}-{ties}/{len(shared)}"

            print(
                f"{task.name:22s} {arm:14s} "
                f"{np.mean(finals):.4g} +- {np.std(finals):<16.4g} "
                f"{np.median(finals):10.4g} {record:>10s} {np.mean(times):6.1f}s"
            )
        print("-" * 96)

    plot(data, tasks)


def plot(data: dict, tasks: list[BatchTask]) -> None:
    """Writes best-so-far trajectories, median across seeds, one panel per task."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [t for t in tasks if t.name in data]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), squeeze=False)
    for ax, task in zip(axes[0], present):
        for arm in ARMS:
            runs = [r["trajectory"] for r in data[task.name].get(arm, {}).values() if r["trajectory"]]
            if not runs:
                continue
            length = min(len(r) for r in runs)
            stacked = np.array([r[:length] for r in runs])
            ax.plot(np.arange(1, length + 1), np.median(stacked, axis=0), label=arm, lw=1.5)
            ax.fill_between(
                np.arange(1, length + 1),
                np.quantile(stacked, 0.25, axis=0),
                np.quantile(stacked, 0.75, axis=0),
                alpha=0.15,
            )
        ax.set_yscale("log")
        ax.set_xlabel("trials")
        ax.set_ylabel("best cost so far")
        ax.set_title(task.name)
        ax.legend()

    fig.tight_layout()
    out = Path("report/batch_selector_trajectory.png")
    fig.savefig(out, dpi=130)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
