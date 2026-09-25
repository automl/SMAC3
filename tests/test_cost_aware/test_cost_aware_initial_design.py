from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac import BlackBoxFacade
from smac.initial_design.cost_aware_initial_design import CostAwareInitialDesign
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.runhistory.dataclasses import TrialValue
from smac.scenario import Scenario


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------

def evaluate_config(config: Configuration) -> dict[str, float]:
    """2D function with a constant performance loss and a four-peak cost landscape.

    The flat performance surface lets the test focus purely on verifying that the
    cost-aware initial design explores the cost surface correctly.
    """
    x, y = config["x"], config["y"]

    cost_unnormalized = (
        np.exp(-((x - 2) ** 2 + (y - 2) ** 2))
        + np.exp(-((x + 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x - 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x + 2) ** 2 + (y - 2) ** 2))
    )
    # Normalise to [0.1, 1.1]
    cost = (cost_unnormalized + 1) / 2 + 0.1
    return {"performance": 1.0, "cost": cost}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=2)
    cs.add(UniformFloatHyperparameter("x", -3.5, 3.5, default_value=0))
    cs.add(UniformFloatHyperparameter("y", -3.5, 3.5, default_value=0))
    return cs


@pytest.fixture
def scenario(configspace: ConfigurationSpace, tmp_path) -> Scenario:
    return Scenario(
        configspace=configspace,
        name="CostAwareInitialDesignTest",
        objectives="cost",
        n_trials=np.inf,
        seed=2,
        deterministic=True,
        output_directory=tmp_path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cost_aware_initial_design_runs_within_budget(scenario, configspace):
    """Cost-aware initial design should stay within the given resource budget."""
    total_resource_budget = 3.0  # small: cost per eval in [0.1, 1.1], ~3-30 evals max

    cost_formula = lambda cfg: evaluate_config(cfg)["cost"]
    cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=cost_formula)

    initial_design = CostAwareInitialDesign(
        scenario=scenario,
        cost_model=cost_model,
        initial_budget=total_resource_budget,
        candidate_pool_size=50,  # small pool to keep the test fast
    )

    smac = BlackBoxFacade(
        scenario=scenario,
        initial_design=initial_design,
        overwrite=True,
    )

    cumulative_cost = 0.0
    iterations = 0

    while cumulative_cost < total_resource_budget:
        trial_info = smac.ask()
        if trial_info is None:
            break

        result = evaluate_config(trial_info.config)
        performance, cost = result["performance"], result["cost"]

        if cumulative_cost + cost > total_resource_budget:
            break

        cumulative_cost += cost
        iterations += 1
        smac.tell(trial_info, TrialValue(cost=performance, time=cost))

    # At least one trial was evaluated
    assert iterations >= 1, "No trials were evaluated"

    # Budget was respected
    assert (
        cumulative_cost <= total_resource_budget + 1e-9
    ), f"Cumulative cost {cumulative_cost:.4f} exceeded budget {total_resource_budget}"

    # The number of finished trials in run history must match the completed iterations.
    # Note: len(smac.runhistory) may be iterations + 1 because smac.ask() registers
    # a RUNNING trial in runhistory before the budget check breaks the loop.
    assert smac.runhistory.finished == iterations
    assert len(smac.runhistory) in (iterations, iterations + 1)


def test_cost_aware_initial_design_evaluates_low_cost_configs(scenario, configspace):
    """Configurations selected by the cost-aware initial design should have
    below-average cost compared to uniform random sampling.
    """
    total_resource_budget = 3.0

    cost_formula = lambda cfg: evaluate_config(cfg)["cost"]
    cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=cost_formula)

    initial_design = CostAwareInitialDesign(
        scenario=scenario,
        cost_model=cost_model,
        initial_budget=total_resource_budget,
        candidate_pool_size=50,
    )

    smac = BlackBoxFacade(
        scenario=scenario,
        initial_design=initial_design,
        overwrite=True,
    )

    evaluated_costs = []
    cumulative_cost = 0.0

    while cumulative_cost < total_resource_budget:
        trial_info = smac.ask()
        if trial_info is None:
            break
        result = evaluate_config(trial_info.config)
        performance, cost = result["performance"], result["cost"]
        if cumulative_cost + cost > total_resource_budget:
            break
        cumulative_cost += cost
        evaluated_costs.append(cost)
        smac.tell(trial_info, TrialValue(cost=performance, time=cost))

    # Sanity: the feature exists (costs are tracked as positive numbers)
    assert len(evaluated_costs) >= 1
    assert all(c > 0 for c in evaluated_costs), "All evaluation costs should be positive"

    # The cost landscape range is [0.1, 1.1]; evaluated costs should be finite
    assert all(
        c <= 1.1 + 1e-9 for c in evaluated_costs
    ), "Evaluated cost exceeds maximum possible cost of the landscape"

    # Selected configurations should have lower mean cost than uniform random sampling
    random_configs = configspace.sample_configuration(size=100)
    random_mean_cost = float(np.mean([evaluate_config(c)["cost"] for c in random_configs]))
    cost_aware_mean_cost = float(np.mean(evaluated_costs))
    assert cost_aware_mean_cost < random_mean_cost, (
        f"Cost-aware mean cost ({cost_aware_mean_cost:.4f}) should be lower than "
        f"random sampling mean cost ({random_mean_cost:.4f})"
    )
