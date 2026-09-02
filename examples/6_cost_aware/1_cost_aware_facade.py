from __future__ import annotations

import logging

import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac.facade.cost_aware_facade import CostAwareFacade
from smac.scenario import Scenario

# Configure logging
logging.basicConfig(level=logging.INFO)


def evaluate_config(config: Configuration, seed: int = 0) -> dict[str, float]:
    """
    A 2D target function where cost and performance are non-trivial.
    - Performance is a simple quadratic function.
    - Cost is a complex landscape.
    """
    x, y = config["x"], config["y"]

    # Performance is a simple bowl shape, minimum at (-2,-1)
    performance = (x+2)**2 + (y+1)**2

    # Cost is a function with four peaks/valleys
    cost_unnormalized = (
        np.exp(-((x - 2) ** 2 + (y - 2) ** 2))
        + np.exp(-((x + 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x - 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x + 2) ** 2 + (y - 2) ** 2))
    )
    # Normalize cost to be in a reasonable range (e.g., [0.1, 1.1])
    cost = (cost_unnormalized + 1) / 2 + 0.1

    return {"performance": performance, "cost": cost}


if __name__ == "__main__":
    # 1. Define the Configuration Space
    configspace = ConfigurationSpace(seed=0)
    configspace.add(UniformFloatHyperparameter("x", -3.5, 3.5, default_value=0))
    configspace.add(UniformFloatHyperparameter("y", -3.5, 3.5, default_value=0))

    # 2. Define the Scenario
    scenario = Scenario(
        configspace=configspace,
        name="EICoolTestWithHandCraftedCost",
        objectives="cost",  # We are minimizing performance loss
        n_trials=np.inf,  # Set high, budget will stop us
        seed=0,
        deterministic=True,
    )

    # 3. Define the cost formula
    cost_formula = lambda config: evaluate_config(config)["cost"]

    # 4. Initialize SMAC facade
    # The facade will automatically handle the initial design and acquisition function.
    smac = CostAwareFacade(
        scenario=scenario,
        target_function=evaluate_config,
        total_resource_budget=50.0,
        cost_formula=cost_formula,
        overwrite=True,
    )

    # 5. --- Optimize ---
    smac.optimize()

    # 6. --- Plot the results ---
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
        if "Initial Design" in config.origin:
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
    plt.savefig("ei_cool_landscapes.svg")
    plt.show()