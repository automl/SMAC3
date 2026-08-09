from __future__ import annotations

import logging
import time

import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac import BlackBoxFacade
from smac.initial_design.cost_aware_initial_design import CostAwareInitialDesign
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.runhistory.dataclasses import TrialValue
from smac.scenario import Scenario
from smac.intensifier.intensifier import Intensifier

# Configure logging
logging.basicConfig(level=logging.INFO)


def evaluate_config(config: Configuration) -> dict[str, float]:
    """
    A 2D target function where the cost is a complex landscape and the loss is constant.
    This allows us to visualize how the initial design explores the cost surface.
    """
    x, y = config["x"], config["y"]

    # The cost is a function with four peaks/valleys
    cost_unnormalized = (
        np.exp(-((x - 2) ** 2 + (y - 2) ** 2))
        + np.exp(-((x + 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x - 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x + 2) ** 2 + (y - 2) ** 2))
    )
    # Normalize cost to be in a reasonable range (e.g., [0.1, 1.1])
    cost = (cost_unnormalized + 1) / 2 + 0.1

    # The performance loss is constant, so optimization focuses purely on exploring
    performance_loss = 1.0

    return {"performance": performance_loss, "cost": cost}


if __name__ == "__main__":
    # 1. Define the Configuration Space
    configspace = ConfigurationSpace()
    configspace.add(UniformFloatHyperparameter("x", -3.5, 3.5, default_value=0))
    configspace.add(UniformFloatHyperparameter("y", -3.5, 3.5, default_value=0))

    # 2. --- Budget Definition ---
    total_resource_budget = 10.0
    initial_design_budget = total_resource_budget 

    # 3. Define the Scenario
    scenario = Scenario(
        configspace=configspace,
        name="CostAwareInitialDesignTest",
        objectives="cost",
        n_trials=np.inf,  # Set high, budget will stop us
        seed=2,
        deterministic=True,
    )

    # 4. Define the cost formula and create the HandCrafted Cost Model
    cost_formula = lambda config: evaluate_config(config)["cost"]
    cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=cost_formula)

    # 5. Create the Cost-Aware Initial Design
    initial_design = CostAwareInitialDesign(
        scenario=scenario,
        cost_model=cost_model,
        initial_budget=initial_design_budget,
        candidate_pool_size=1000
    )

    # 6. Initialize SMAC facade, passing our custom initial design
    smac = BlackBoxFacade(
        scenario=scenario,
        initial_design=initial_design,
        overwrite=True
    )

    # 7. --- Standard Budget-Based Optimization Loop ---
    cumulative_cost = 0.0
    print("\n--- Starting Budget-Based Optimization Loop ---")
    while cumulative_cost < total_resource_budget:
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

        value = TrialValue(cost=performance, time=cost)
        smac.tell(trial_info, value)

    print("\n--- Total resource budget exhausted. ---")

    # 8. --- Plot the results ---
    # Create a grid for the contour plot
    grid_res = 100
    x_grid = np.linspace(-3.5, 3.5, grid_res)
    y_grid = np.linspace(-3.5, 3.5, grid_res)
    xx, yy = np.meshgrid(x_grid, y_grid)
    cost_grid = np.zeros_like(xx)
    for i in range(grid_res):
        for j in range(grid_res):
            cost_grid[i, j] = evaluate_config(Configuration(configspace, {"x": xx[i, j], "y": yy[i, j]}))["cost"]

    # Get evaluated points from runhistory
    initial_x, initial_y = [], []
    bo_x, bo_y = [], []
    all_costs = []
    for k, v in smac.runhistory.items():
        config = smac.runhistory.get_config(k.config_id)
        # The origin is automatically set by the initial design class
        if config.origin == "Cost Aware Initial Design":
            initial_x.append(config["x"])
            initial_y.append(config["y"])
        else:
            bo_x.append(config["x"])
            bo_y.append(config["y"])
        all_costs.append(v.time)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("CostAwareInitialDesign Class Test", fontsize=16)

    # Plot Cost Landscape
    contour = ax1.contourf(xx, yy, cost_grid, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax1, label="Cost")
    ax1.scatter(initial_x, initial_y, c="blue", edgecolor="white", s=80, label="Cost Aware Initial Design Points", zorder=2)
    #ax1.scatter(bo_x, bo_y, c="red", marker="X", edgecolor="white", s=80, label="BO Points", zorder=2)
    ax1.set_title("Cost Landscape with Evaluated Points")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # Plot Cumulative Cost
    cumulative_cost_plot = np.cumsum(all_costs)
    ax2.plot(range(len(cumulative_cost_plot)), cumulative_cost_plot, "-o", label="Cumulative Cost")
    ax2.axhline(y=initial_design_budget, color="orange", linestyle="--", label="Initial Design Budget")
    ax2.axhline(y=total_resource_budget, color="r", linestyle="--", label="Total Budget")
    ax2.set_title("Cumulative Cost Over Trials")
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("Cumulative Cost")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("cost_aware_initial_design_class_test.png")
    plt.show()