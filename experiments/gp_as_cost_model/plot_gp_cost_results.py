from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
from smac.runhistory import RunHistory
from smac.scenario import Scenario

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Seaborn Setup ---
sns.set_theme(style="whitegrid", font_scale=1.2)
palette = sns.color_palette("deep")


def evaluate_config(config: Configuration) -> dict[str, float]:
    """Re-definition of the target function for plotting the ground truth."""
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
    # 1. Load data from the SMAC run
    # The actual output is in a subdirectory named after the seed
    base_output_dir = "smac3_output/GP_Cost_Model_Test"
    seed = 0
    run_output_dir = os.path.join(base_output_dir, str(seed))

    # Load configspace and runhistory manually from the correct paths
    configspace = ConfigurationSpace(seed=seed)
    configspace.add(UniformFloatHyperparameter("x", -3.5, 3.5, default_value=0))
    configspace.add(UniformFloatHyperparameter("y", -3.5, 3.5, default_value=0))

    runhistory = RunHistory()
    runhistory.load(os.path.join(run_output_dir, "runhistory.json"), configspace)
    
    cost_model_path = os.path.join(run_output_dir, "cost_model.pkl")
    with open(cost_model_path, "rb") as f:
        cost_model = pickle.load(f)
    print(f"Loaded cost model from {cost_model_path}")

    # 2. Prepare data for plotting
    grid_res = 100
    x_grid = np.linspace(-3.5, 3.5, grid_res)
    y_grid = np.linspace(-3.5, 3.5, grid_res)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Ground truth grids
    perf_grid = np.zeros_like(xx)
    true_cost_grid = np.zeros_like(xx)
    grid_configs = []
    for i in range(grid_res):
        for j in range(grid_res):
            config = Configuration(configspace, {"x": xx[i, j], "y": yy[i, j]})
            result = evaluate_config(config)
            perf_grid[i, j] = result["performance"]
            true_cost_grid[i, j] = result["cost"]
            grid_configs.append(config)

    # Learned GP cost grid
    grid_arrays = np.array([c.get_array() for c in grid_configs])
    pred_mean, pred_var = cost_model.predict(grid_arrays)
    pred_mean_grid = pred_mean.reshape(grid_res, grid_res)
    pred_std_grid = np.sqrt(pred_var.reshape(grid_res, grid_res))

    # Evaluated points
    points = {
        "sampling": {"x": [], "y": []},
        "cost_aware_initial": {"x": [], "y": []},
        "bo": {"x": [], "y": []},
    }
    for k, v in runhistory.items():
        config = runhistory.get_config(k.config_id)
        origin = config.origin
        key = "bo"  # Default
        if origin == "Sampling":
            key = "sampling"
        elif origin == "Cost Aware Initial Design":
            key = "cost_aware_initial"
        
        points[key]["x"].append(config["x"])
        points[key]["y"].append(config["y"])

    # 3. Create plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle("Cost-Aware Optimization with GP Cost Model", fontsize=20)
    (ax1, ax2), (ax3, ax4) = axes

    # Plot 1: Performance Landscape
    c1 = ax1.contourf(xx, yy, perf_grid, levels=20, cmap="magma")
    fig.colorbar(c1, ax=ax1, label="Performance Loss")
    ax1.set_title("Ground Truth Performance")

    # Plot 2: True Cost Landscape
    c2 = ax2.contourf(xx, yy, true_cost_grid, levels=20, cmap="viridis")
    fig.colorbar(c2, ax=ax2, label="Evaluation Cost")
    ax2.set_title("Ground Truth Cost")

    # Plot 3: Learned Cost Landscape (GP Mean)
    c3 = ax3.contourf(xx, yy, pred_mean_grid, levels=20, cmap="viridis")
    fig.colorbar(c3, ax=ax3, label="Predicted Mean Log-Scaled Cost")
    ax3.set_title("Learned Cost Model (GP Mean)")

    # Plot 4: Learned Cost Uncertainty (GP Std Dev)
    c4 = ax4.contourf(xx, yy, pred_std_grid, levels=20, cmap="cividis")
    fig.colorbar(c4, ax=ax4, label="Predicted Std. Dev.")
    ax4.set_title("Learned Cost Model (GP Uncertainty)")

    # Define the true optimum of the performance function
    optimum_x, optimum_y = -2, -1

    # Overlay evaluated points on all plots
    for ax in axes.flatten():
        ax.scatter(points["sampling"]["x"], points["sampling"]["y"], color=palette[1], marker='D', s=80, ec="k", label="Bootstrap", zorder=3)
        ax.scatter(points["cost_aware_initial"]["x"], points["cost_aware_initial"]["y"], color=palette[0], s=80, ec="k", label="Initial Design", zorder=2)
        ax.scatter(points["bo"]["x"], points["bo"]["y"], color=palette[3], marker="X", s=100, ec="k", label="BO Points", zorder=2)
        ax.scatter(optimum_x, optimum_y, marker='*', color='gold', s=200, ec='black', label='Perf. Optimum', zorder=4)
        ax.set_xlabel("x"), ax.set_ylabel("y"), ax.legend(loc="upper right", framealpha=0.5), ax.grid(True, alpha=0.5)

    # Use tight_layout with padding arguments for better control
    plt.tight_layout(pad=3.0, h_pad=4.0)
    
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, "experiment_figures")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "gp_cost_model_results.pdf"))
    plt.show()