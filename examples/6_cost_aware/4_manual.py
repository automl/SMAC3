from __future__ import annotations

import logging
import time
from typing import Iterator

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
from scipy.spatial.distance import cdist

from smac import BlackBoxFacade
from smac.acquisition.function.cost_aware_acquisition_function import CostAwareAcquisitionFunction
from smac.acquisition.function.expected_improvement import EI
from smac.callback.cost_surrogate_callback import CostSurrogateCallback
from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.model.random_forest import RandomForest
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.scenario import Scenario

# Configure logging
logging.basicConfig(level=logging.INFO)


# Helper class: Custom Empty Initial Design so SMAC does not attempt to evaluate
# additional default initial design configurations when smac.ask() is first called.
class EmptyInitialDesign(AbstractInitialDesign):
    def _select_configurations(self) -> list[Configuration]:
        return []

    def select_configurations(self) -> Iterator[Configuration]:
        return iter([])


# 1. Primary evaluation function that returns performance loss and resource cost
def evaluate_config(config: Configuration) -> dict[str, float]:
    """Evaluates a configuration and returns both its performance loss and resource cost."""
    x = config["x"]
    resource_cost = 0.1 + abs(x - 3) / 5
    performance_loss = (x - 5) ** 2
    return {"performance": performance_loss, "cost": resource_cost}


if __name__ == "__main__":
    # Define the Configuration Space
    configspace = ConfigurationSpace()
    configspace.add(UniformFloatHyperparameter("x", 0, 10, default_value=5))

    # --- Budget Definition ---
    total_resource_budget = 20.0  # Total evaluation resource budget
    initial_design_budget = total_resource_budget / 8.0  # Budget fraction for initial design
    # -------------------------

    # Define the Scenario (deterministic=True avoids re-evaluating configs on multiple seeds)
    scenario = Scenario(
        configspace=configspace,
        name="CostAwareManualExample",
        objectives="cost",
        n_trials=1000,  # Set high, total_resource_budget will stop optimization
        seed=42,
        deterministic=True,
    )

    # --- Manual Component Setup ---

    # A. Create surrogate cost model to predict evaluation costs
    cost_model = RandomForest(configspace=configspace, seed=scenario.seed)

    # B. Create cost surrogate callback to retrain cost_model after each trial
    cost_surrogate_callback = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    # C. Create cost-aware acquisition function (EI-Cool strategy)
    acquisition_function = CostAwareAcquisitionFunction(
        acquisition_function=EI(),
        cost_surrogate_callback=cost_surrogate_callback,
    )

    # D. Instantiate BlackBoxFacade with EmptyInitialDesign so it relies purely on our manual initial design
    smac = BlackBoxFacade(
        scenario=scenario,
        initial_design=EmptyInitialDesign(scenario),
        acquisition_function=acquisition_function,
        callbacks=[cost_surrogate_callback],
        overwrite=True,
    )

    # --- Manual Warm-Start and Budget-Based Optimization Loop ---
    cumulative_cost = 0.0
    candidate_pool = configspace.sample_configuration(size=1000)
    selected_configs_manual: list[Configuration] = []

    print("\n--- Starting Manual Initial Design and Optimization Loop ---")
    while cumulative_cost < total_resource_budget:
        # --- Phase 1: Manual Cost-Aware Initial Design ---
        if cumulative_cost < initial_design_budget:
            if not selected_configs_manual:
                # Bootstrap: Select the first point randomly to train the cost model initial state
                chosen_config = candidate_pool[0]
            else:
                # Cost-aware candidate elimination (Algorithm 1)
                pruning_candidates = list(candidate_pool)
                candidate_arrays = np.array([c.get_array() for c in pruning_candidates])
                costs, _ = cost_model.predict(candidate_arrays)
                costs = costs.flatten()

                selected_arrays = np.array([c.get_array() for c in selected_configs_manual])
                indices = list(range(len(pruning_candidates)))

                while len(indices) > 1:
                    # Remove the most expensive candidate
                    max_cost_local_idx = np.argmax(costs[indices])
                    indices.pop(max_cost_local_idx)

                    if len(indices) == 1:
                        break

                    # Remove the candidate closest to already-selected points
                    current_arrays = candidate_arrays[indices]
                    distances = cdist(current_arrays, selected_arrays)
                    closest_local_idx = np.argmin(np.min(distances, axis=1))
                    indices.pop(closest_local_idx)

                chosen_config = pruning_candidates[indices[0]]

            chosen_config.origin = "Manual Cost-Aware Initial Design"
            trial_info = TrialInfo(config=chosen_config, seed=scenario.seed)
            candidate_pool.remove(chosen_config)
            selected_configs_manual.append(chosen_config)

        # --- Phase 2: Bayesian Optimization (EI-Cool) ---
        else:
            if len(selected_configs_manual) > 0:
                print("\n--- Initial Design Budget Exhausted. Switching to Bayesian Optimization ---")
                selected_configs_manual = []  # Clear marker

            # Update cost-aware acquisition function with current budget info before ask()
            acquisition_function.set_budget_info(
                total_budget=total_resource_budget,
                cumulative_cost=cumulative_cost,
                initial_design_budget=initial_design_budget,
            )

            # Ask SMAC for the next configuration
            trial_info = smac.ask()

        # Perform evaluation
        result = evaluate_config(trial_info.config)
        performance, cost = result["performance"], result["cost"]

        # Check if evaluation exceeds total budget
        if cumulative_cost + cost > total_resource_budget:
            print(f"Evaluation cost ({cost:.2f}) would exceed total budget ({total_resource_budget:.2f}). Stopping.")
            break

        cumulative_cost += cost
        print(
            f"Origin: {trial_info.config.origin}, Cost: {cost:.2f}, "
            f"Cumulative Cost: {cumulative_cost:.2f}/{total_resource_budget:.2f}"
        )

        # Tell SMAC evaluation results (cost_surrogate_callback retrains cost_model automatically)
        value = TrialValue(
            cost=performance,
            time=cost,
            starttime=time.time(),
            endtime=time.time() + cost,
            additional_info={"resource_cost": cost},
        )
        smac.tell(trial_info, value)

    print("\n--- Total resource budget exhausted. ---")

    # Retrieve best configuration found
    incumbent = smac.intensifier.get_incumbent()
    if incumbent is not None:
        print(f"\nBest configuration found: {incumbent}")
        res = evaluate_config(incumbent)
        print(f"Validated performance loss: {res['performance']:.4f}")
        print(f"Validated resource cost: {res['cost']:.4f}")
    else:
        print("No incumbent found.")