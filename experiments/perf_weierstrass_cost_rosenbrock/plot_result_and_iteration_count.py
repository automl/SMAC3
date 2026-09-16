import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
from smac.runhistory import RunHistory
import ioh
import seaborn as sns

# --- Configuration ---
SCENARIO_COST_AWARE = "Weierstrass_Perf_Rosenbrock_Cost_Aware"
SCENARIO_RANDOM_EI = "Weierstrass_Perf_Rosenbrock_Random_EI"
N_SEEDS = 10
TOTAL_BUDGET = 10000.0

# --- Seaborn Setup ---
sns.set_theme(style="whitegrid", font_scale=1.2)
palette = sns.color_palette("deep")
cost_aware_color = palette[0]  # Blue
random_ei_color = palette[3]   # Red

def load_and_process_data(scenario_name: str, configspace: ConfigurationSpace):
    """Loads all runhistories for a scenario and processes cost, performance, and best config data."""
    all_runs_costs = []
    all_best_configs = []

    for seed in range(N_SEEDS):
        runhistory_path = os.path.join(f"smac3_output/{scenario_name}", str(seed), "runhistory.json")
        if not os.path.exists(runhistory_path):
            print(f"Warning: Runhistory not found for seed {seed} in {scenario_name}. Skipping.")
            continue

        rh = RunHistory()
        rh.load(runhistory_path, configspace)
        
        sorted_runs = sorted(rh.items(), key=lambda item: item[1].starttime)
        
        run_costs = [v.time for k, v in sorted_runs]
        all_runs_costs.append(run_costs)

        best_perf_for_seed = float('inf')
        best_config_for_seed = None
        for k, v in sorted_runs:
            if v.cost < best_perf_for_seed:
                best_perf_for_seed = v.cost
                best_config_for_seed = rh.get_config(k.config_id)
        
        if best_config_for_seed:
            all_best_configs.append([best_config_for_seed["x0"], best_config_for_seed["x1"]])

    if not all_runs_costs:
        return None, None

    cost_df = pd.DataFrame(all_runs_costs).transpose()
    cost_data = (cost_df.median(axis=1), cost_df.quantile(0.25, axis=1), cost_df.quantile(0.75, axis=1), cost_df.cumsum().median(axis=1))

    return cost_data, all_best_configs

# --- Load Data for Both Experiments ---
configspace = ConfigurationSpace(seed=0)
configspace.add(UniformFloatHyperparameter("x0", -5, 5, default_value=0))
configspace.add(UniformFloatHyperparameter("x1", -5, 5, default_value=0))

cost_aware_data, best_configs_ca = load_and_process_data(SCENARIO_COST_AWARE, configspace)
random_ei_data, best_configs_rei = load_and_process_data(SCENARIO_RANDOM_EI, configspace)

if not cost_aware_data or not random_ei_data:
    raise FileNotFoundError("Could not load data for one or both experiments. Please run them first.")

# --- Load Data for a single run for plotting path ---
single_run_rh_ca = RunHistory()
single_run_path_ca = os.path.join(f"smac3_output/{SCENARIO_COST_AWARE}", "0", "runhistory.json")
bootstrap_points, initial_design_points, bo_points = [], [], []
if os.path.exists(single_run_path_ca):
    single_run_rh_ca.load(single_run_path_ca, configspace)
    sorted_single_run_ca = sorted(single_run_rh_ca.items(), key=lambda item: item[1].starttime)
    
    for k, v in sorted_single_run_ca:
        config = single_run_rh_ca.get_config(k.config_id)
        point = (config["x0"], config["x1"])
        if config.origin == "Sampling":
            bootstrap_points.append(point)
        elif config.origin == "Cost Aware Initial Design":
            initial_design_points.append(point)
        else:
            bo_points.append(point)
else:
    print("Warning: Could not find single runhistory for cost-aware variant to plot path.")



# --- Plotting ---
fig = plt.figure(figsize=(24, 16))

# --- Top Row: Landscape Plots ---
# Re-define functions to generate grid data
weierstrass = ioh.get_problem(16, instance=1, dimension=2, problem_class=ioh.ProblemClass.BBOB)
weierstrass_optimum = weierstrass.optimum.x
rosenbrock = ioh.get_problem(8, instance=1, dimension=2, problem_class=ioh.ProblemClass.BBOB)
rosenbrock_optimum = rosenbrock.optimum.x

def evaluate_config(config: Configuration): 
    x = [config["x0"], config["x1"]]
    return {"performance": weierstrass(x), "cost": rosenbrock(x)}

grid_res = 100
x_grid, y_grid = np.linspace(-5, 5, grid_res), np.linspace(-5, 5, grid_res)
xx, yy = np.meshgrid(x_grid, y_grid)
perf_grid, cost_grid = np.zeros_like(xx), np.zeros_like(xx)
for i in range(grid_res):
    for j in range(grid_res):
        result = evaluate_config(Configuration(configspace, {"x0": xx[i, j], "x1": yy[i, j]}))
        perf_grid[i, j] = result["performance"]
        cost_grid[i, j] = result["cost"]

# Plot 1: Performance Landscape (Weierstrass)
ax1 = fig.add_subplot(2, 2, 1)
contour_perf = ax1.contourf(xx, yy, perf_grid, levels=20, cmap="viridis")
fig.colorbar(contour_perf, ax=ax1, label="Weierstrass Value")
ax1.scatter([weierstrass_optimum[0]], [weierstrass_optimum[1]], marker='*', s=400, color='yellow', edgecolor='black', label='Weierstrass Optimum', zorder=15)

if best_configs_ca:
    best_x_ca, best_y_ca = zip(*best_configs_ca)
    ax1.scatter(best_x_ca, best_y_ca, color=cost_aware_color, marker='D', s=100, edgecolor='white', linewidth=1.5, label='Best Found (Cost-Aware)', zorder=12)
if best_configs_rei:
    best_x_rei, best_y_rei = zip(*best_configs_rei)
    ax1.scatter(best_x_rei, best_y_rei, color=random_ei_color, marker='s', s=100, edgecolor='white', linewidth=1.5, label='Best Found (Random+EI)', zorder=11)

ax1.set_title("Performance Landscape (Weierstrass)"), ax1.set_xlabel("x0"), ax1.set_ylabel("x1")
legend1 = ax1.legend()
legend1.set_zorder(20)
ax1.grid(True, alpha=0.3)

# Plot 2: Cost Landscape (Rosenbrock)
ax2 = fig.add_subplot(2, 2, 2)
contour_cost = ax2.contourf(xx, yy, np.log(cost_grid + 1e-9), levels=20, cmap="magma")
fig.colorbar(contour_cost, ax=ax2, label="Log(Rosenbrock Cost)")
ax2.scatter([rosenbrock_optimum[0]], [rosenbrock_optimum[1]], marker='*', s=400, color='yellow', edgecolor='black', label='Rosenbrock Minimum', zorder=15)

# Plot single run path for cost-aware
if bootstrap_points:
    boot_x, boot_y = zip(*bootstrap_points)
    ax2.scatter(boot_x, boot_y, color=palette[1], marker='D', s=80, ec="k", label="Bootstrap", zorder=11)

if initial_design_points:
    init_x, init_y = zip(*initial_design_points)
    ax2.scatter(init_x, init_y, color=palette[0], marker='o', s=80, ec="k", label="Initial Design", zorder=10)

if bo_points:
    bo_x, bo_y = zip(*bo_points)
    ax2.scatter(bo_x, bo_y, color=palette[3], marker="X", s=100, ec="k", label="BO Points", zorder=12)


ax2.set_title("Cost Landscape (Rosenbrock) for single cost-aware run"), ax2.set_xlabel("x0"), ax2.set_ylabel("x1")
legend2 = ax2.legend(loc="lower right", framealpha=0.7)
legend2.set_zorder(20)
ax2.grid(True, alpha=0.3)

# --- Bottom Row: Combined Cost Plot ---
ax3 = fig.add_subplot(2, 1, 2)
median_ca, q25_ca, q75_ca, cumsum_ca = cost_aware_data
ax3.plot(median_ca.index, median_ca, color=cost_aware_color, label='Cost-Aware', marker='.', markersize=8, zorder=10)
ax3.fill_between(median_ca.index, q25_ca, q75_ca, color=cost_aware_color, alpha=0.2)
early_phase_ca_indices = cumsum_ca[cumsum_ca <= TOTAL_BUDGET / 8].index
ax3.plot(early_phase_ca_indices, median_ca.loc[early_phase_ca_indices], color=cost_aware_color, marker='*', markersize=15, linestyle='None', markeredgecolor='black', label='Cost-Aware (Early Phase)', zorder=11)

median_rei, q25_rei, q75_rei, cumsum_rei = random_ei_data
ax3.plot(median_rei.index, median_rei, color=random_ei_color, label='Random+EI', marker='.', markersize=8, zorder=5)
ax3.fill_between(median_rei.index, q25_rei, q75_rei, color=random_ei_color, alpha=0.2)
early_phase_rei_indices = cumsum_rei[cumsum_rei <= TOTAL_BUDGET / 8].index
ax3.plot(early_phase_rei_indices, median_rei.loc[early_phase_rei_indices], color=random_ei_color, marker='*', markersize=15, linestyle='None', markeredgecolor='black', label='Random+EI (Early Phase)', zorder=6)

ax3.set_title("Comparison of Median Cost per Trial (Rosenbrock Cost)")
ax3.set_xlabel("Trial Number")
ax3.set_ylabel("Cost")
legend3 = ax3.legend()
legend3.set_zorder(20)
ax3.grid(True, which="both", alpha=0.5), ax3.set_yscale("log")

# --- Finalize and Save ---
plt.tight_layout(pad=3.0)
script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, "experiment_figures")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "weierstrass_perf_rosenbrock_cost_comparison.svg"))
plt.show()