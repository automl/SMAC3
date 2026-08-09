from __future__ import annotations

import logging
import os

import ioh
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac.facade.blackbox_facade import BlackBoxFacade
from smac.initial_design.cost_aware_initial_design import CostAwareInitialDesign
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.runhistory.dataclasses import TrialValue, TrialInfo
from smac.scenario import Scenario

# Configure logging and plot style
logging.basicConfig(level=logging.INFO)
sns.set_theme(style="whitegrid")


def run_ioh_experiment(problem_id: int, total_budget: float):
    """
    Runs a SMAC experiment with CostAwareInitialDesign on a given IOH problem,
    where the evaluation cost is the function's performance value and performance is constant.
    """
    # 1. Set up the IOH problem
    dimension = 2
    problem = ioh.get_problem(problem_id, instance=1, dimension=dimension, problem_class=ioh.ProblemClass.BBOB)
    problem_name = problem.meta_data.name

    # --- Cost Normalization Setup ---
    grid_res = 100
    x_vals = np.linspace(-5, 5, grid_res)
    y_vals = np.linspace(-5, 5, grid_res)
    raw_costs = np.array([problem([x, y]) for x in x_vals for y in y_vals])
    min_raw_cost, max_raw_cost = np.min(raw_costs), np.max(raw_costs)

    def normalize_cost(raw_cost: float) -> float:
        if max_raw_cost == min_raw_cost:
            return 50.0
        return 100 * (raw_cost - min_raw_cost) / (max_raw_cost - min_raw_cost)

    # 2. Define the target function for SMAC
    # Performance is constant, cost is the normalized IOH function value.
    def evaluate_config(config: Configuration, seed: int = 0) -> dict[str, float]:
        x = [config[f"x{i}"] for i in range(dimension)]
        raw_cost = problem(x)
        normalized_cost = normalize_cost(raw_cost)
        return {"performance": 1.0, "cost": normalized_cost}

    # 3. Define the Configuration Space and Scenario
    configspace = ConfigurationSpace(seed=0)
    for i in range(dimension):
        configspace.add(UniformFloatHyperparameter(f"x{i}", -5, 5, default_value=0))

    scenario = Scenario(
        configspace=configspace,
        name=f"CostAwareInitialDesign_on_{problem_name}",
        objectives="cost",
        n_trials=np.inf,
        seed=0,
        deterministic=True,
    )

    # 4. Set up Cost-Aware Components
    cost_formula = lambda config: evaluate_config(config)["cost"]
    cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=cost_formula)
    
    # We need a runhistory to track the costs
    from smac.runhistory.runhistory import RunHistory
    runhistory = RunHistory()

    initial_design = CostAwareInitialDesign(
        scenario=scenario,
        cost_model=cost_model,
        initial_budget=total_budget,
        runhistory=runhistory,
    )

    # 5. Initialize SMAC with BlackBoxFacade
    smac = BlackBoxFacade(
        scenario=scenario,
        initial_design=initial_design,
        runhistory=runhistory,
        overwrite=True,
    )

    print(f"\n--- Testing Cost-Aware Initial Design on {problem_name} ---")
    for i in range(int(total_budget) * 2):  # Loop more times than expected trials
        trial_info = smac.ask()

        if trial_info is None:
            print("SMAC has no more configurations to suggest.")
            break

        if "Initial Design" not in trial_info.config.origin and "Sampling" not in trial_info.config.origin:
            print("Initial design phase is complete. Stopping.")
            break

        result = evaluate_config(trial_info.config)
        performance, cost = result["performance"], result["cost"]
        smac.tell(trial_info, TrialValue(cost=performance, time=cost))
        print(f"Trial {len(smac.runhistory)} (Origin: {trial_info.config.origin}): Cost={cost:.2f}")

    print("--- Initial design test finished. ---")

    # 7. Plotting
    plot_results(smac, problem_name, evaluate_config)


def plot_results(smac: BlackBoxFacade, problem_name: str, evaluate_config: callable):
    """Generates and saves plots for the experiment results."""
    configspace = smac.scenario.configspace
    runhistory = smac.runhistory

    # Create a grid for the cost contour plot
    grid_res = 100
    x_grid = np.linspace(-5, 5, grid_res)
    y_grid = np.linspace(-5, 5, grid_res)
    xx, yy = np.meshgrid(x_grid, y_grid)
    cost_grid = np.zeros_like(xx)
    for i in range(grid_res):
        for j in range(grid_res):
            config = Configuration(configspace, {"x0": xx[i, j], "x1": yy[i, j]})
            cost_grid[i, j] = evaluate_config(config)["cost"]

    # --- FIX: Filter runhistory to only include points from the initial design ---
    initial_design_trials = [
        (k, v) for k, v in runhistory.items() if "Cost Aware Initial Design" in runhistory.get_config(k.config_id).origin
    ]
    sorted_trials = sorted(initial_design_trials, key=lambda item: item[1].starttime)
    
    if not sorted_trials:
        print("No points from initial design found in runhistory. Cannot plot.")
        return

    points_x = [runhistory.get_config(k.config_id)["x0"] for k, v in sorted_trials]
    points_y = [runhistory.get_config(k.config_id)["x1"] for k, v in sorted_trials]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle(f"Cost-Aware Initial Design on {problem_name}", fontsize=18, y=0.95)

    # Cost Landscape with new colormap
    contour = ax.contourf(xx, yy, cost_grid, levels=30, cmap="magma")
    fig.colorbar(contour, ax=ax, label="Normalized Cost [0-100]")
    
    # Scatter plot with improved visibility
    ax.scatter(
        points_x, 
        points_y, 
        marker='o',
        c="cyan", 
        edgecolor="black", 
        linewidth=1.5,
        s=200, 
        label="Evaluated Points", 
        zorder=3
    )
    
    # Add numbers to the points to show the evaluation order
    for i, (px, py) in enumerate(zip(points_x, points_y)):
        ax.text(px, py, str(i+1), color="black", ha="center", va="center", fontsize=8, fontweight='bold')

    ax.set_title("Cost Landscape with Evaluation Order")
    ax.set_xlabel("x0"), ax.set_ylabel("x1"), ax.legend()
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save the plot in the same directory as the script
    plt.savefig(os.path.join(script_dir, f"initial_design_test_{problem_name.lower()}_normalized_cost.svg"), dpi=150)
    plt.show()


if __name__ == "__main__":
    # The budget is now in terms of the normalized cost (0-100 scale)
    BUDGET = 500.0

    # We use different budgets for different functions due to their cost characteristics, 
    # so we can have almost the same number of evaluations

    # --- Run experiment for Sphere function (BBOB f1) ---
    run_ioh_experiment(problem_id=1, total_budget=BUDGET)

    # --- Run experiment for Rosenbrock function (BBOB f8) ---
    run_ioh_experiment(problem_id=8, total_budget=BUDGET/20)

    # --- Run experiment for Weierstrass function (BBOB f16) ---
    run_ioh_experiment(problem_id=16, total_budget=BUDGET/5)