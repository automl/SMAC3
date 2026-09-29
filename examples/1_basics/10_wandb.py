"""Example of using SMAC with Weights & Biases.

To run this example, install the optional W&B dependency with ``pip install smac[wandb]``. 
WandbRunner logs each SMAC target-function evaluation as a separate W&B run 
while SMAC remains responsible for the optimization.
"""


from __future__ import annotations

import argparse

import numpy as np

from ConfigSpace import Configuration, ConfigurationSpace, Float

from smac import HyperparameterOptimizationFacade, Scenario
from smac.runner.wandb_runner import WandbRunner


class Branin:
    def __init__(self, seed: int = 0):
        cs = ConfigurationSpace(seed=seed)
        x0 = Float("x0", (-5, 10), default=-5, log=False)
        x1 = Float("x1", (0, 15), default=2, log=False)
        cs.add([x0, x1])

        self.cs = cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        x0 = config["x0"]
        x1 = config["x1"]
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)
        return a * (x1 - b * x0**2 + c * x0 - r) ** 2 \
             + s * (1 - t) * np.cos(x0) + s


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize the Branin function with SMAC and log trials to W&B."
    )

    parser.add_argument(
        "--entity",
        type=str,
        required=True,
        help="W&B entity to which the runs will be logged.",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="W&B project to which the runs will be logged.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    model = Branin()

    # Define the SMAC scenario as usual. W&B does not change the optimization setup or configuration space.
    scenario = Scenario(
        configspace=model.cs,
        deterministic=True,
        n_trials=10,
        trial_walltime_limit=100,
        n_workers=1,
    )

    # Wrap the target function with WandbRunner to log each SMAC trial as a separate W&B run.
    runner = WandbRunner(
        scenario=scenario,
        target_function=model.train,
        entity=args.entity,
        project=args.project,
    )

    # Use the runner as SMAC's target function. SMAC remains responsible for selecting configurations and optimization.
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=runner,
    )

    incumbent = smac.optimize()

    print(f"Best configuration: {incumbent}")
