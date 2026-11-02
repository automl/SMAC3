"""Example of using SMAC with Weights & Biases.

Install the optional dependency with ``pip install smac[wandb]``.

WandbRunner logs objective values and custom trial metrics to individual
W&B runs. WandbCallback logs optimization-level metrics to a separate
run in the same W&B project. SMAC remains responsible for optimization.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from smac import HyperparameterOptimizationFacade, Scenario
from smac.callback.wandb_callback import WandbCallback
from smac.main.smbo import SMBO
from smac.runner.wandb_runner import WandbRunner


__copyright__ = "Copyright 2025, Leibniz University Hannover, Institute of AI"
__license__ = "3-clause BSD"


class DigitsMLP:
    """MLP classifier with a configurable architecture and training setup."""

    def __init__(self, seed: int = 0):
        self.cs = ConfigurationSpace(seed=seed)
        self.cs.add(
            [
                Integer("n_layers", (1, 3), default=2),
                Integer("n_neurons", (16, 128), log=True, default=64),
                Categorical("activation", ["relu", "tanh"], default="relu"),
                Float("alpha", (1e-5, 1e-1), log=True, default=1e-3),
                Float(
                    "learning_rate_init",
                    (1e-4, 1e-2),
                    log=True,
                    default=1e-3,
                ),
            ]
        )

        digits = load_digits()
        self.X_train, self.X_validation, self.y_train, self.y_validation = (
            train_test_split(
                digits.data,
                digits.target,
                test_size=0.25,
                stratify=digits.target,
                random_state=seed,
            )
        )

    def train(
        self,
        config: Configuration, 
        seed: int,
    ) -> tuple[float, dict[str, float]]:
        """Train and evaluate an MLP.

        Returns the objective value separately from additional metrics.
        WandbRunner logs both to the corresponding trial run, while SMAC
        uses the objective value to guide the optimization.
        """
        classifier = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(
                    (config["n_neurons"],) * config["n_layers"]
                ),
                activation=config["activation"],
                alpha=config["alpha"],
                learning_rate_init=config["learning_rate_init"],
                max_iter=150,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=seed,
            ),
        )

        start_time = time.perf_counter()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            classifier.fit(self.X_train, self.y_train)

        training_time = time.perf_counter() - start_time

        predictions = classifier.predict(self.X_validation)
        accuracy = accuracy_score(self.y_validation, predictions)
        error = 1.0 - accuracy

        # Additional information is logged to W&B alongside the objective.
        # It does not become part of SMAC's optimization objective.
        return float(error), {
            "training_time_seconds": float(training_time),
            "n_iterations": float(classifier[-1].n_iter_),
        }


def log_optimization_metrics(smbo: SMBO) -> dict[str, float]:
    """Return metrics describing the progress of the optimization.

    This function is called by WandbCallback after a trial result has been
    added to SMAC's RunHistory. It logs to a separate W&B run, allowing
    users to monitor the optimization as a whole rather than individual
    model-training evaluations.
    """
    costs = [
        float(value.cost)
        for value in smbo.runhistory.values()
        if value.cost is not None
        and np.isscalar(value.cost)
        and np.isfinite(value.cost)
    ]

    metrics = {"n_evaluations": float(len(costs))}

    if costs:
        best_error = min(costs)
        metrics["best_validation_error"] = best_error

    return metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize an MLP with SMAC and log metrics to W&B."
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
    model = DigitsMLP()

    scenario = Scenario(
        configspace=model.cs,
        deterministic=True,
        n_trials=20,
    )

    # Wrap the target function with WandbRunner to track every evaluation as an individual W&B run.
    runner = WandbRunner(
        scenario=scenario,
        target_function=model.train,
        entity=args.entity,
        project=args.project,
        required_arguments=["seed"],
    )

    # The callback tracks optimization-level metrics in a separate W&B run.
    #
    # Providing a logging_function is optional. If no custom optimization
    # metrics are needed, the WandbRunner can be used without explicitly
    # supplying a WandbCallback.
    callback = WandbCallback(
        runner=runner,
        logging_function=log_optimization_metrics,
    )

    # Pass the runner as the facade's target function and register the
    # callback to enable custom optimization-level tracking.
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=runner,
        callbacks=[callback],
        overwrite=True,
    )

    incumbent = smac.optimize()

    print(f"Best configuration: {incumbent}")