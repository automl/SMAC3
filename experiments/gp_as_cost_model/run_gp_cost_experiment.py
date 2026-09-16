from __future__ import annotations

import logging
import os
import pickle

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac.facade.cost_aware_facade import CostAwareFacade
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.scenario import Scenario

# Configure logging
logging.basicConfig(level=logging.INFO)


def evaluate_config(config: Configuration, seed: int = 0) -> dict[str, float]:
    """
    A 2D target function where cost and performance are non-trivial.
    """
    x, y = config["x"], config["y"]
    performance_loss = (x + 2) ** 2 + (y + 1) ** 2
    cost_unnormalized = (
        np.exp(-((x - 2) ** 2 + (y - 2) ** 2))
        + np.exp(-((x + 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x - 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x + 2) ** 2 + (y - 2) ** 2))
    )
    cost = (cost_unnormalized + 1) / 2 + 0.1
    return {"performance": performance_loss, "cost": cost}


if __name__ == "__main__":
    # 1. Define the Configuration Space and Scenario
    configspace = ConfigurationSpace(seed=0)
    configspace.add(UniformFloatHyperparameter("x", -3.5, 3.5, default_value=0))
    configspace.add(UniformFloatHyperparameter("y", -3.5, 3.5, default_value=0))

    scenario = Scenario(
        configspace=configspace,
        name="GP_Cost_Model_Test",
        objectives="cost",
        n_trials=np.inf,
        seed=0,
        deterministic=True,
    )

    # 2. Define the Cost Model (Gaussian Process)
    cost_model = BlackBoxFacade.get_model(scenario=scenario)

    # 3. Initialize and run SMAC
    total_budget = 50.0

    smac = CostAwareFacade(
        scenario=scenario,
        target_function=evaluate_config,
        total_resource_budget=total_budget,
        cost_model=cost_model,
        overwrite=True,
    )

    smac.optimize()

    # 5. Save the final trained cost model for plotting
    output_dir = smac.scenario.output_directory
    cost_model_path = os.path.join(output_dir, "cost_model.pkl")
    with open(cost_model_path, "wb") as f:
        pickle.dump(cost_model, f)
    print(f"Final cost model saved to {cost_model_path}")