"""Expected Improvement per Second (EIPS)
# Flags: doc-Runnable

An example of applying SMAC with Expected Improvement per Second (EIPS).

EIPS extends Expected Improvement by considering the predicted runtime of a
configuration. Instead of maximizing only the expected improvement, EIPS
maximizes the expected improvement per unit of time:

    EIPS(x) = EI(x) / E[T(x)]

This is useful when evaluation times vary substantially between
configurations. By modeling both objective values and runtimes, EIPS balances
expected improvement against evaluation cost, favoring configurations that are
expected to make the most progress per unit of evaluation time.

Although EIPS selects candidate configurations based on their expected improvement per unit of evaluation time, 
the optimization objective itself remains unchanged. 
SMAC still searches for the configuration with the best objective value; 
runtime is only used to guide the search more efficiently.

Internally, SMAC requires three components to use EIPS:

* ``RunHistoryEIPSEncoder``
    Encodes objective values and runtimes.

* ``EIPS``
    Acquisition function that combines expected improvement and predicted
    runtime.

* ``MultiObjectiveModel``
    Maintains separate surrogate models for the objective and runtime.
    The surrogate models can be chosen independently.
"""


from __future__ import annotations

import time

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt

from smac import RunHistory, Scenario
from smac.acquisition.function.expected_improvement import EIPS
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade as HPOFacade,
)
from smac.model.gaussian_process.gaussian_process import GaussianProcess
from smac.model.random_forest.random_forest import RandomForest
from smac.model.multi_objective_model import MultiObjectiveModel
from smac.runhistory.encoder.eips_encoder import RunHistoryEIPSEncoder

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class ExpensiveQuadratic:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        cs.add(Float("x", (-10, 10), default=-10))
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Quadratic objective with a varying evaluation runtime.

        The runtime varies across the configuration space independently of the
        objective value, making it suitable for demonstrating Expected Improvement
        per Second (EIPS).
        """

        x = config["x"]

        # Simulated runtime
        runtime = 1 + np.cos(2 * np.pi * x / 4)
        time.sleep(runtime)

        return x ** 2


def plot(runhistory: RunHistory, incumbent: Configuration) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(7, 6))

    xs = np.linspace(-10, 10, 400)

    # Ground-truth objective
    ax[0].plot(xs, xs ** 2, label="Objective")

    for k, v in runhistory.items():
        cfg = runhistory.get_config(k.config_id)
        ax[0].scatter(cfg["x"], v.cost, alpha=0.3)

    ax[0].scatter(
        incumbent["x"],
        incumbent["x"] ** 2,
        color="red",
        marker="x",
        s=100,
        label="Incumbent",
    )

    ax[0].set_ylabel("Objective")
    ax[0].legend()

    # Ground-truth runtime
    runtime = 1 + np.cos(2 * np.pi * xs / 4)
    ax[1].plot(xs, runtime)

    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Runtime [s]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = ExpensiveQuadratic()

    scenario = Scenario(
        model.configspace,
        deterministic=True,
        n_trials=60,
    )

    # EIPS requires an encoder that stores both objective values and runtimes.
    encoder = RunHistoryEIPSEncoder(scenario)

    # The EIPS acquisition function.
    acquisition_function = EIPS()

    # Here, we use a GP for the objective and a Random Forest for the runtime,
    # as runtime models are often less smooth than objective functions.
    surrogate = MultiObjectiveModel(
        models=[
            GaussianProcess(model.configspace, BlackBoxFacade.get_kernel(scenario)), # Objective
            RandomForest(model.configspace) # Runtime
        ],
        objectives=["cost", "time"],
    )

    smac = HPOFacade(
        scenario,
        model.train,
        overwrite=True,
        runhistory_encoder=encoder,
        acquisition_function=acquisition_function,
        model=surrogate,
    )

    incumbent = smac.optimize()

    default_cost = smac.validate(model.configspace.get_default_configuration())
    incumbent_cost = smac.validate(incumbent)

    print(f"Default cost: {default_cost:.4f}")
    print(f"Incumbent cost: {incumbent_cost:.4f}")

    plot(smac.runhistory, incumbent)