from __future__ import annotations

import logging
import time

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
from scipy.spatial.distance import cdist

from smac.callback.cost_surrogate_callback import CostSurrogateCallback
from smac import BlackBoxFacade
from smac.model.random_forest import RandomForest
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.scenario import Scenario


# Configure logging
logging.basicConfig(level=logging.INFO)


# 1. Create a primary evaluation function that returns both performance and cost
def evaluate_config(config: Configuration) -> dict[str, float]:
    """
    Evaluates a configuration and returns both its performance loss and resource cost.
    """
    x = config["x"]
    resource_cost = 0.1 + abs(x - 3) / 5
    performance_loss = (x - 5) ** 2
    return {"performance": performance_loss, "cost": resource_cost}


if __name__ == "__main__":
    # Define the Configuration Space
    configspace = ConfigurationSpace()
    configspace.add(UniformFloatHyperparameter("x", 0, 10, default_value=5))

    # --- Budget Definition ---
    total_resource_budget = 20.0  # seconds
    initial_design_budget = total_resource_budget / 8.0
    # -------------------------

    # Define the Scenario
    scenario = Scenario(
        configspace=configspace,
        name="CostAwareExample",
        objectives="cost",
        n_trials=1000,  # Set high, budget will stop us
        seed=42
    )

    # Create the Cost Model
    cost_model = RandomForest(configspace=configspace, seed=scenario.seed)
    cost_surrogate_callback = CostSurrogateCallback(cost_model=cost_model)

    # Initialize SMAC facade.
    smac = BlackBoxFacade(
        scenario=scenario,
        callbacks=[cost_surrogate_callback],
        overwrite=True,
        initial_design=BlackBoxFacade.get_initial_design(
            scenario,
            n_configs=0,  # Do not use the default initial design at all
        ),
    )

    # --- Manual Warm-Start and Budget-Based Optimization Loop ---
    cumulative_cost = 0.0
    candidate_pool = configspace.sample_configuration(size=1000)
    selected_configs_manual: list[Configuration] = []

    print("\n--- Starting Manual Initial Design and Optimization Loop ---")
    while cumulative_cost < total_resource_budget:
        # --- Phase-Based Configuration Selection ---
        if cumulative_cost < initial_design_budget:
            # --- Manual Initial Design Phase ---
            if not selected_configs_manual:
                # Bootstrap: Select the first point randomly
                chosen_config = candidate_pool[0]
            else:
                # Pruning loop based on Algorithm 1
                pruning_candidates = list(candidate_pool)
                while len(pruning_candidates) > 1:
                    candidate_arrays = np.array([c.get_array() for c in pruning_candidates])
                    costs, _ = cost_model.predict(candidate_arrays)
                    max_cost_idx = np.argmax(costs)
                    pruning_candidates.pop(max_cost_idx)

                    if len(pruning_candidates) == 1:
                        break

                    selected_arrays = np.array([c.get_array() for c in selected_configs_manual])
                    candidate_arrays = np.array([c.get_array() for c in pruning_candidates])
                    distances = cdist(candidate_arrays, selected_arrays)
                    closest_idx = np.argmin(np.min(distances, axis=1))
                    pruning_candidates.pop(closest_idx)
                chosen_config = pruning_candidates[0]

            chosen_config.origin = "Manual Initial Design"
            trial_info = TrialInfo(config=chosen_config, seed=scenario.seed)
            candidate_pool.remove(chosen_config)
            selected_configs_manual.append(chosen_config)
        else:
            # --- Bayesian Optimization Phase ---
            if len(selected_configs_manual) > 0:
                print("\n--- Initial Design Budget Exhausted. Switching to Bayesian Optimization ---")
                selected_configs_manual = []  # Clear to prevent this message from repeating

            trial_info = smac.ask()
        # -----------------------------------------

        # Perform a real evaluation
        result = evaluate_config(trial_info.config)
        performance, cost = result["performance"], result["cost"]

        # Check if this evaluation would exceed the total budget
        if cumulative_cost + cost > total_resource_budget:
            print(f"Evaluation cost ({cost:.2f}) would exceed total budget. Stopping.")
            break

        cumulative_cost += cost
        print(f"Origin: {trial_info.config.origin}, Cost: {cost:.2f}, "
              f"Cumulative Cost: {cumulative_cost:.2f}/{total_resource_budget:.2f}")

        # Tell SMAC the results. The callback will train the cost_model automatically.
        value = TrialValue(
            cost=performance,
            time=cost,
            starttime=time.time(),
            endtime=time.time() + cost,
        )
        smac.tell(trial_info, value)

    print("\n--- Total resource budget exhausted. ---")
    # -----------------------------------------

    # Get the incumbent from the intensifier
    incumbent = smac.intensifier.get_incumbent()
    if incumbent is not None:
        print(f"Best configuration found: {incumbent}")
        performance = evaluate_config(incumbent)["performance"]
        print(f"Validated cost: {performance}")
    else:
        print("No incumbent found.")