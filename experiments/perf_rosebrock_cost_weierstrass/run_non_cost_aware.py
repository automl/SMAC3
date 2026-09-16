import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
from smac.facade.cost_aware_facade import CostAwareFacade
from smac.scenario import Scenario
from smac.initial_design import RandomInitialDesign
from smac.acquisition.function import EI
import ioh

def run_experiment(seed: int):
    """Runs the non-cost-aware (Random+EI) optimization."""
    print(f"--- Starting WEIERSTRASS RANDOM+EI run for seed {seed} ---")

    # --- Experiment Setup (Identical to cost-aware script) ---
    rosenbrock = ioh.get_problem(8, instance=1, dimension=2, problem_class=ioh.ProblemClass.BBOB)
    rosenbrock_optimum = rosenbrock.optimum.x
    weierstrass = ioh.get_problem(16, instance=1, dimension=2, problem_class=ioh.ProblemClass.BBOB)
    cost_center = rosenbrock_optimum - np.array([5.0, 5.0])
    cost_center = np.clip(cost_center, -5.0, 5.0)
    shift = weierstrass.optimum.x - cost_center

    def weierstrass_cost_func(x: list[float]) -> float:
        return weierstrass(np.array(x) + shift)

    configspace = ConfigurationSpace(seed=seed)
    configspace.add(UniformFloatHyperparameter("x0", -5, 5, default_value=0))
    configspace.add(UniformFloatHyperparameter("x1", -5, 5, default_value=0))

    def evaluate_config(config: Configuration, seed: int = 0):
        x = [config["x0"], config["x1"]]
        return {"performance": rosenbrock(x), "cost": weierstrass_cost_func(x)}

    # --- SMAC Setup ---
    scenario = Scenario(
        configspace=configspace,
        name="Weierstrass_Random_EI",
        objectives="cost",
        n_trials=np.inf,
        seed=seed,
        deterministic=True,
    )

    initial_design_instance = RandomInitialDesign(scenario=scenario, n_configs=5)
    acquisition_function_instance = EI()

    smac = CostAwareFacade(
        scenario=scenario,
        target_function=evaluate_config,
        total_resource_budget=10000.0,
        cost_formula=lambda config: weierstrass_cost_func([config["x0"], config["x1"]]),
        initial_design=initial_design_instance,
        acquisition_function=acquisition_function_instance
    )

    # --- Run Optimization ---
    print(f"Running optimization. Results will be saved in {smac.scenario.output_directory}")
    smac.optimize()
    print(f"--- Optimization for seed {seed} finished. ---")

if __name__ == "__main__":
    N_SEEDS = 10
    for i in range(N_SEEDS):
        run_experiment(seed=i)