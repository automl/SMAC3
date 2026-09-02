from __future__ import annotations

import logging

import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac import BlackBoxFacade
from smac.acquisition.function.cost_aware_acquisition_function import CostAwareAcquisitionFunction
from smac.acquisition.function.expected_improvement import EI
from smac.callback.cost_surrogate_callback import CostSurrogateCallback
from smac.initial_design.cost_aware_initial_design import CostAwareInitialDesign
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.runhistory.dataclasses import TrialValue
from smac.scenario import Scenario

# Configure logging
logging.basicConfig(level=logging.INFO)


def evaluate_config(config: Configuration) -> dict[str, float]:
    """
    A 2D target function where cost and performance are non-trivial.
    - Performance is a simple quadratic function.
    - Cost is a complex landscape.
    """
    x, y = config["x"], config["y"]

    # Performance is a simple bowl shape, minimum at (-2,-1)
    performance_loss = (x+2)**2 + (y+1)**2

    # Cost is a function with four peaks/valleys
    cost_unnormalized = (
        np.exp(-((x - 2) ** 2 + (y - 2) ** 2))
        + np.exp(-((x + 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x - 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x + 2) ** 2 + (y - 2) ** 2))
    )
    # Normalize cost to be in a reasonable range (e.g., [0.1, 1.1])
    cost = (cost_unnormalized + 1) / 2 + 0.1

    return {"performance": performance_loss, "cost": cost}


if __name__ == "__main__":
    # 1. Define the Configuration Space
    configspace = ConfigurationSpace(seed=0)
    configspace.add(UniformFloatHyperparameter("x", -3.5, 3.5, default_value=0))
    configspace.add(UniformFloatHyperparameter("y", -3.5, 3.5, default_value=0))

    # 2. --- Budget Definition ---
    total_resource_budget = 50.0
    initial_design_budget = 0

    # 3. Define the Scenario
    scenario = Scenario(
        configspace=configspace,
        name="EICoolTestWithHandCraftedCost",
        objectives="cost",  # We are minimizing performance loss
        n_trials=np.inf,  # Set high, budget will stop us
        seed=0,
        deterministic=True,
    )

    # 4. Define the Cost Model and its callback
    cost_formula = lambda config: evaluate_config(config)["cost"]
    cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=cost_formula)
    cost_surrogate_callback = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    # 5. Create the Cost-Aware Initial Design
    initial_design = CostAwareInitialDesign(
        scenario=scenario,
        cost_model=cost_model,
        initial_budget=initial_design_budget,
        candidate_pool_size=1000,
    )

    # 6. Define the EI-Cool acquisition function
    base_acquisition_function = EI()
    acquisition_function = CostAwareAcquisitionFunction(
        acquisition_function=base_acquisition_function,
        cost_surrogate_callback=cost_surrogate_callback
    )

    # 7. Initialize SMAC facade, passing our custom components
    smac = BlackBoxFacade(
        scenario=scenario,
        initial_design=initial_design,
        acquisition_function=acquisition_function,
        callbacks=[cost_surrogate_callback],
        overwrite=True,
    )

    # 8. --- Budget-Based Optimization Loop ---
    cumulative_cost = initial_design_budget
    print("\n--- Starting Budget-Based Optimization Loop with EI-Cool ---")

    # Main optimization loop
    while cumulative_cost < total_resource_budget:
        # Set the budget info on our acquisition function directly
        acquisition_function.set_budget_info(
            total_budget=total_resource_budget,
            cumulative_cost=cumulative_cost,
            initial_design_budget=initial_design_budget,
        )

        trial_info = smac.ask()

        if trial_info is None:
            print("SMAC has no more configurations to suggest. Stopping.")
            break

        result = evaluate_config(trial_info.config)
        performance, cost = result["performance"], result["cost"]

        if cumulative_cost + cost > total_resource_budget:
            print(f"Evaluation cost ({cost:.2f}) would exceed total budget. Stopping.")
            break

        cumulative_cost += cost
        print(f"Origin: {trial_info.config.origin}, Cost: {cost:.2f}, "
              f"Cumulative Cost: {cumulative_cost:.2f}/{total_resource_budget:.2f}")

        smac.tell(trial_info, TrialValue(cost=performance, time=cost))

    print("\n--- Total resource budget exhausted. ---")

    # 9. --- Plot the results ---
    grid_res = 100
    x_grid = np.linspace(-3.5, 3.5, grid_res)
    y_grid = np.linspace(-3.5, 3.5, grid_res)
    xx, yy = np.meshgrid(x_grid, y_grid)
    perf_grid = np.zeros_like(xx)
    cost_grid = np.zeros_like(xx)
    for i in range(grid_res):
        for j in range(grid_res):
            result = evaluate_config(Configuration(configspace, {"x": xx[i, j], "y": yy[i, j]}))
            perf_grid[i, j] = result["performance"]
            cost_grid[i, j] = result["cost"]

    initial_x, initial_y, bo_x, bo_y = [], [], [], []
    for k, v in smac.runhistory.items():
        config = smac.runhistory.get_config(k.config_id)
        if config.origin == "Cost Aware Initial Design":
            initial_x.append(config["x"])
            initial_y.append(config["y"])
        else:
            bo_x.append(config["x"])
            bo_y.append(config["y"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("EI-Cool with Cost-Aware Initial Design (Imperfect Cost Model)", fontsize=16)

    # Plot Performance Landscape
    contour1 = ax1.contourf(xx, yy, perf_grid, levels=20, cmap="magma")
    fig.colorbar(contour1, ax=ax1, label="Performance Loss")
    ax1.scatter(initial_x, initial_y, c="blue", edgecolor="white", s=80, label="Cost Aware Initial Design Points", zorder=2)
    ax1.scatter(bo_x, bo_y, c="red", marker="X", edgecolor="white", s=100, label="BO Points (EI-Cool)", zorder=2)
    ax1.set_title("Performance Landscape with Evaluated Points")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # Plot Cost Landscape
    contour2 = ax2.contourf(xx, yy, cost_grid, levels=20, cmap="viridis")
    fig.colorbar(contour2, ax=ax2, label="Evaluation Cost")
    ax2.scatter(initial_x, initial_y, c="blue", edgecolor="white", s=80, label="Initial Design Points", zorder=2)
    ax2.scatter(bo_x, bo_y, c="red", marker="X", edgecolor="white", s=100, label="BO Points (EI-Cool)", zorder=2)
    ax2.set_title("Cost Landscape with Evaluated Points")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("ei_cool_landscapes.png")
    plt.show()