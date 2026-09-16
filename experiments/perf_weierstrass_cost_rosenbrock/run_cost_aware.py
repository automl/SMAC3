import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
from smac.facade.cost_aware_facade import CostAwareFacade
from smac.scenario import Scenario
import ioh

def run_experiment(seed: int):
    """Runs the cost-aware optimization with Weierstrass as performance and Rosenbrock as cost."""
    print(f"--- Starting WEIERSTRASS-PERF/ROSENBROCK-COST (Cost-Aware) run for seed {seed} ---")

    # --- Experiment Setup ---
    # 1. Performance Function: Weierstrass (BBOB ID=16)
    weierstrass = ioh.get_problem(16, instance=1, dimension=2, problem_class=ioh.ProblemClass.BBOB)

    # 2. Cost Function: Rosenbrock (BBOB ID=8)
    rosenbrock = ioh.get_problem(8, instance=1, dimension=2, problem_class=ioh.ProblemClass.BBOB)

    def normalized_rosenbrock_cost(x: list[float]) -> float:
        """
        Calculates Rosenbrock value and normalizes it to the range [100, 1000].
        Assumes raw Rosenbrock values are in [0, 100000] for the [-5, 5] domain.
        """
        raw_cost = rosenbrock(x)
        
        # Min-max scaling formula: new_min + (value - old_min) * (new_range / old_range)
        old_min, old_max = 0, 100000  # Estimated range of raw Rosenbrock cost
        new_min, new_max = 100, 1000  # Desired range for normalized cost
        
        # Clip the raw cost to prevent values outside the expected range
        clipped_cost = np.clip(raw_cost, old_min, old_max)
        
        normalized_cost = new_min + (clipped_cost - old_min) * (new_max - new_min) / (old_max - old_min)
        return normalized_cost

    # 3. Configuration Space
    configspace = ConfigurationSpace(seed=seed)
    configspace.add(UniformFloatHyperparameter("x0", -5, 5, default_value=0))
    configspace.add(UniformFloatHyperparameter("x1", -5, 5, default_value=0))

    # 4. Target Evaluation Function
    def evaluate_config(config: Configuration, seed: int = 0):
        x = [config["x0"], config["x1"]]
        performance = weierstrass(x)
        cost = normalized_rosenbrock_cost(x)
        return {"performance": performance, "cost": cost}

    # 5. SMAC Scenario and Facade
    scenario = Scenario(
        configspace=configspace,
        name="Weierstrass_Perf_Rosenbrock_Cost_Aware",
        objectives="cost",  # This refers to the performance value (Weierstrass)
        n_trials=np.inf,
        seed=seed,
        deterministic=True,
    )

    smac = CostAwareFacade(
        scenario=scenario,
        target_function=evaluate_config,
        total_resource_budget=10000.0,
        cost_formula=lambda config: normalized_rosenbrock_cost([config["x0"], config["x1"]]),
        overwrite=True,
    )

    # --- Run Optimization ---
    print(f"Running optimization. Results will be saved in {smac.scenario.output_directory}")
    smac.optimize()
    print(f"--- Optimization for seed {seed} finished. ---")

if __name__ == "__main__":
    N_SEEDS = 10
    for i in range(N_SEEDS):
        run_experiment(seed=i)