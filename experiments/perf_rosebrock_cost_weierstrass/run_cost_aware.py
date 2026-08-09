import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
from smac.facade.cost_aware_facade import CostAwareFacade
from smac.scenario import Scenario
import ioh

def run_experiment(seed: int):
    """Runs the cost-aware optimization using Weierstrass as the cost function."""
    print(f"--- Starting WEIERSTRASS COST-AWARE run for seed {seed} ---")

    # --- Experiment Setup ---
    # 1. Performance Function: Rosenbrock (BBOB ID=8)
    rosenbrock = ioh.get_problem(8, instance=1, dimension=2, problem_class=ioh.ProblemClass.BBOB)
    rosenbrock_optimum = rosenbrock.optimum.x

    # 2. Cost Function: Weierstrass (BBOB ID=16), shifted to make Rosenbrock optimum expensive
    weierstrass = ioh.get_problem(16, instance=1, dimension=2, problem_class=ioh.ProblemClass.BBOB)
    cost_center = rosenbrock_optimum - np.array([5.0, 5.0])
    cost_center = np.clip(cost_center, -5.0, 5.0)
    # The input to weierstrass is shifted so its minimum moves to cost_center
    shift = weierstrass.optimum.x - cost_center

    def weierstrass_cost_func(x: list[float]) -> float:
        return weierstrass(np.array(x) + shift)

    # 3. Configuration Space
    configspace = ConfigurationSpace(seed=seed)
    configspace.add(UniformFloatHyperparameter("x0", -5, 5, default_value=0))
    configspace.add(UniformFloatHyperparameter("x1", -5, 5, default_value=0))

    # 4. Target Evaluation Function
    def evaluate_config(config: Configuration, seed: int = 0):
        x = [config["x0"], config["x1"]]
        return {"performance": rosenbrock(x), "cost": weierstrass_cost_func(x)}

    # 5. SMAC Scenario and Facade
    scenario = Scenario(
        configspace=configspace,
        name="Weierstrass_Cost_Aware",
        objectives="cost",
        n_trials=np.inf,
        seed=seed,
        deterministic=True,
    )

    smac = CostAwareFacade(
        scenario=scenario,
        target_function=evaluate_config,
        total_resource_budget=10000.0,
        cost_formula=lambda config: weierstrass_cost_func([config["x0"], config["x1"]])
    )

    # --- Run Optimization ---
    print(f"Running optimization. Results will be saved in {smac.scenario.output_directory}")
    smac.optimize()
    print(f"--- Optimization for seed {seed} finished. ---")

if __name__ == "__main__":
    N_SEEDS = 10
    for i in range(N_SEEDS):
        run_experiment(seed=i)