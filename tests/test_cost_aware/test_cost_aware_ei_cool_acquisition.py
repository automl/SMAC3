from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac import BlackBoxFacade
from smac.acquisition.function.cost_aware_acquisition_function import CostAwareAcquisitionFunction
from smac.acquisition.function.expected_improvement import EI
from smac.callback.cost_surrogate_callback import CostSurrogateCallback
from smac.initial_design.cost_aware_initial_design import CostAwareInitialDesign
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.runhistory.dataclasses import StatusType, TrialValue
from smac.scenario import Scenario

# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------

def evaluate_config(config: Configuration) -> dict[str, float]:
    """2D function with a bowl-shaped performance landscape and a four-peak cost landscape.

    Performance minimum at (-2, -1); cost landscape is deliberately misaligned to
    create a non-trivial trade-off between cheap and good regions.
    """
    x, y = config["x"], config["y"]

    performance_loss = (x + 2) ** 2 + (y + 1) ** 2

    cost_unnormalized = (
        np.exp(-((x - 2) ** 2 + (y - 2) ** 2))
        + np.exp(-((x + 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x - 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x + 2) ** 2 + (y - 2) ** 2))
    )
    # Normalise to [0.1, 1.1]
    cost = (cost_unnormalized + 1) / 2 + 0.1

    return {"performance": performance_loss, "cost": cost}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)
    cs.add(UniformFloatHyperparameter("x", -3.5, 3.5, default_value=0))
    cs.add(UniformFloatHyperparameter("y", -3.5, 3.5, default_value=0))
    return cs


@pytest.fixture
def scenario(configspace: ConfigurationSpace, tmp_path) -> Scenario:
    return Scenario(
        configspace=configspace,
        name="EICoolTestWithHandCraftedCost",
        objectives="cost",
        n_trials=np.inf,
        seed=0,
        deterministic=True,
        output_directory=tmp_path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ei_cool_loop_runs_within_budget(scenario, configspace):
    """EI-Cool acquisition loop should terminate without error and respect the budget."""
    total_resource_budget = 3.0  # cost per eval in [0.1, 1.1], keeps CI fast
    initial_design_budget = 0.0

    cost_formula = lambda cfg: evaluate_config(cfg)["cost"]
    cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=cost_formula)
    cost_surrogate_callback = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    initial_design = CostAwareInitialDesign(
        scenario=scenario,
        cost_model=cost_model,
        initial_budget=initial_design_budget,
        candidate_pool_size=50,
    )

    acquisition_function = CostAwareAcquisitionFunction(
        acquisition_function=EI(),
        cost_surrogate_callback=cost_surrogate_callback,
    )

    smac = BlackBoxFacade(
        scenario=scenario,
        initial_design=initial_design,
        acquisition_function=acquisition_function,
        callbacks=[cost_surrogate_callback],
        overwrite=True,
    )

    cumulative_cost = initial_design_budget
    iterations = 0

    while cumulative_cost < total_resource_budget:
        acquisition_function.set_budget_info(
            total_budget=total_resource_budget,
            cumulative_cost=cumulative_cost,
            initial_design_budget=initial_design_budget,
        )

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

    assert iterations >= 1, "No trials were evaluated"
    assert cumulative_cost <= total_resource_budget + 1e-9
    assert smac.runhistory.finished == iterations
    assert len(smac.runhistory) in (iterations, iterations + 1)


def test_ei_cool_budget_info_is_passed_before_each_ask(scenario, configspace):
    """set_budget_info must be called before every ask(); this test verifies that
    the budget parameters and alpha are updated correctly on each iteration.
    """
    total_resource_budget = 3.0
    initial_design_budget = 0.0

    cost_formula = lambda cfg: evaluate_config(cfg)["cost"]
    cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=cost_formula)
    cost_surrogate_callback = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    initial_design = CostAwareInitialDesign(
        scenario=scenario,
        cost_model=cost_model,
        initial_budget=initial_design_budget,
        candidate_pool_size=50,
    )

    acquisition_function = CostAwareAcquisitionFunction(
        acquisition_function=EI(),
        cost_surrogate_callback=cost_surrogate_callback,
    )

    smac = BlackBoxFacade(
        scenario=scenario,
        initial_design=initial_design,
        acquisition_function=acquisition_function,
        callbacks=[cost_surrogate_callback],
        overwrite=True,
    )

    cumulative_cost = initial_design_budget
    recorded_cumulative_costs: list[float] = []
    recorded_alphas: list[float] = []

    while cumulative_cost < total_resource_budget:
        acquisition_function.set_budget_info(
            total_budget=total_resource_budget,
            cumulative_cost=cumulative_cost,
            initial_design_budget=initial_design_budget,
        )
        assert acquisition_function._cumulative_cost == cumulative_cost
        assert acquisition_function._total_budget == total_resource_budget
        recorded_cumulative_costs.append(cumulative_cost)

        trial_info = smac.ask()
        if trial_info is None:
            break

        recorded_alphas.append(acquisition_function._alpha)

        result = evaluate_config(trial_info.config)
        performance, cost = result["performance"], result["cost"]

        if cumulative_cost + cost > total_resource_budget:
            break

        cumulative_cost += cost
        smac.tell(trial_info, TrialValue(cost=performance, time=cost))

    # Costs should be non-decreasing
    for i in range(1, len(recorded_cumulative_costs)):
        assert recorded_cumulative_costs[i] >= recorded_cumulative_costs[i - 1], (
            f"Cumulative cost decreased at iteration {i}: "
            f"{recorded_cumulative_costs[i - 1]:.4f} -> {recorded_cumulative_costs[i]:.4f}"
        )

    # Alpha must decrease towards 0 as budget is consumed
    for i in range(1, len(recorded_alphas)):
        assert (
            recorded_alphas[i] <= recorded_alphas[i - 1] + 1e-9
        ), f"Alpha should decrease as budget is spent: {recorded_alphas[i - 1]:.4f} -> {recorded_alphas[i]:.4f}"


def test_ei_cool_runhistory_records_trial_costs(scenario, configspace):
    """Costs stored in the run history should match the values returned by evaluate_config."""
    total_resource_budget = 2.0
    initial_design_budget = 0.0

    cost_formula = lambda cfg: evaluate_config(cfg)["cost"]
    cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=cost_formula)
    cost_surrogate_callback = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    initial_design = CostAwareInitialDesign(
        scenario=scenario,
        cost_model=cost_model,
        initial_budget=initial_design_budget,
        candidate_pool_size=50,
    )

    acquisition_function = CostAwareAcquisitionFunction(
        acquisition_function=EI(),
        cost_surrogate_callback=cost_surrogate_callback,
    )

    smac = BlackBoxFacade(
        scenario=scenario,
        initial_design=initial_design,
        acquisition_function=acquisition_function,
        callbacks=[cost_surrogate_callback],
        overwrite=True,
    )

    cumulative_cost = initial_design_budget
    told_performances: list[float] = []
    told_costs: list[float] = []

    while cumulative_cost < total_resource_budget:
        acquisition_function.set_budget_info(
            total_budget=total_resource_budget,
            cumulative_cost=cumulative_cost,
            initial_design_budget=initial_design_budget,
        )

        trial_info = smac.ask()
        if trial_info is None:
            break

        result = evaluate_config(trial_info.config)
        performance, cost = result["performance"], result["cost"]

        if cumulative_cost + cost > total_resource_budget:
            break

        cumulative_cost += cost
        told_performances.append(performance)
        told_costs.append(cost)
        smac.tell(
            trial_info,
            TrialValue(cost=performance, time=cost, additional_info={"resource_cost": cost}),
        )

    assert len(told_performances) >= 1, "No trials were completed"
    assert smac.runhistory.finished == len(told_performances)
    assert len(smac.runhistory) in (len(told_performances), len(told_performances) + 1)

    finished_trials = [v for v in smac.runhistory.values() if v.status == StatusType.SUCCESS]
    assert len(finished_trials) == len(told_performances)
    for run_value, expected_perf, expected_cost in zip(finished_trials, told_performances, told_costs):
        assert np.isclose(run_value.cost, expected_perf)
        assert np.isclose(run_value.time, expected_cost)
        assert np.isclose(run_value.additional_info["resource_cost"], expected_cost)
