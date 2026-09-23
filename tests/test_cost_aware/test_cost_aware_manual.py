from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
from scipy.spatial.distance import cdist

from smac import BlackBoxFacade
from smac.acquisition.function.cost_aware_acquisition_function import CostAwareAcquisitionFunction
from smac.acquisition.function.expected_improvement import EI
from smac.callback.cost_surrogate_callback import CostSurrogateCallback
from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.model.random_forest import RandomForest
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.scenario import Scenario


# ---------------------------------------------------------------------------
# Helper: suppress SMAC's own default initial design
# ---------------------------------------------------------------------------

class EmptyInitialDesign(AbstractInitialDesign):
    """Yields no configurations so the manual warm-start loop has full control."""

    def _select_configurations(self) -> list[Configuration]:
        return []

    def select_configurations(self) -> Iterator[Configuration]:
        return iter([])


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------

def evaluate_config(config: Configuration) -> dict[str, float]:
    """1D function: cost increases away from x=3, performance minimum at x=5."""
    x = config["x"]
    resource_cost = 0.1 + abs(x - 3) / 5
    performance_loss = (x - 5) ** 2
    return {"performance": performance_loss, "cost": resource_cost}


# ---------------------------------------------------------------------------
# Core loop helper (shared between tests to avoid duplication)
# ---------------------------------------------------------------------------

def _run_manual_loop(
    smac: BlackBoxFacade,
    acquisition_function: CostAwareAcquisitionFunction,
    cost_model: RandomForest,
    configspace: ConfigurationSpace,
    scenario: Scenario,
    total_resource_budget: float,
    initial_design_budget: float,
    candidate_pool_size: int = 20,
) -> tuple[float, int]:
    """Execute the manual cost-aware loop and return (cumulative_cost, iterations)."""
    cumulative_cost = 0.0
    candidate_pool = list(configspace.sample_configuration(size=candidate_pool_size))
    selected_configs_manual: list[Configuration] = []
    iterations = 0

    while cumulative_cost < total_resource_budget:
        # Phase 1: Manual cost-aware initial design
        if cumulative_cost < initial_design_budget:
            if not selected_configs_manual:
                chosen_config = candidate_pool[0]
            else:
                pruning_candidates = list(candidate_pool)
                candidate_arrays = np.array([c.get_array() for c in pruning_candidates])
                costs, _ = cost_model.predict(candidate_arrays)
                costs = costs.flatten()

                selected_arrays = np.array([c.get_array() for c in selected_configs_manual])
                indices = list(range(len(pruning_candidates)))

                while len(indices) > 1:
                    max_cost_local_idx = np.argmax(costs[indices])
                    indices.pop(max_cost_local_idx)
                    if len(indices) == 1:
                        break
                    current_arrays = candidate_arrays[indices]
                    distances = cdist(current_arrays, selected_arrays)
                    closest_local_idx = np.argmin(np.min(distances, axis=1))
                    indices.pop(closest_local_idx)

                chosen_config = pruning_candidates[indices[0]]

            chosen_config.origin = "Manual Cost-Aware Initial Design"
            trial_info = TrialInfo(config=chosen_config, seed=scenario.seed)
            candidate_pool.remove(chosen_config)
            selected_configs_manual.append(chosen_config)

        # Phase 2: Bayesian optimisation (EI-Cool)
        else:
            if selected_configs_manual:
                selected_configs_manual = []

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

        smac.tell(
            trial_info,
            TrialValue(
                cost=performance,
                time=cost,
                starttime=time.time(),
                endtime=time.time() + cost,
                additional_info={"resource_cost": cost},
            ),
        )

    return cumulative_cost, iterations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=42)
    cs.add(UniformFloatHyperparameter("x", 0, 10, default_value=5))
    return cs


@pytest.fixture
def scenario(configspace: ConfigurationSpace, tmp_path) -> Scenario:
    return Scenario(
        configspace=configspace,
        name="CostAwareManualTest",
        objectives="cost",
        n_trials=100,  # generous ceiling; budget stops the loop first
        seed=42,
        deterministic=True,
        output_directory=tmp_path,
    )


@pytest.fixture
def cost_model_and_callback(configspace, scenario):
    cost_model = RandomForest(configspace=configspace, seed=scenario.seed)
    callback = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)
    return cost_model, callback


@pytest.fixture
def smac_facade(scenario, cost_model_and_callback):
    _, cost_surrogate_callback = cost_model_and_callback
    acquisition_function = CostAwareAcquisitionFunction(
        acquisition_function=EI(),
        cost_surrogate_callback=cost_surrogate_callback,
    )
    smac = BlackBoxFacade(
        scenario=scenario,
        initial_design=EmptyInitialDesign(scenario),
        acquisition_function=acquisition_function,
        callbacks=[cost_surrogate_callback],
        overwrite=True,
    )
    return smac, acquisition_function


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_manual_loop_runs_within_budget(scenario, configspace, cost_model_and_callback, smac_facade):
    """Manual loop must terminate without error and stay within the resource budget."""
    total_resource_budget = 2.0   # cost per eval in [0.1, 0.9] for x in [0,10], keeps CI fast
    initial_design_budget = total_resource_budget / 4.0

    cost_model, cost_surrogate_callback = cost_model_and_callback
    smac, acquisition_function = smac_facade

    cumulative_cost, iterations = _run_manual_loop(
        smac=smac,
        acquisition_function=acquisition_function,
        cost_model=cost_model,
        configspace=configspace,
        scenario=scenario,
        total_resource_budget=total_resource_budget,
        initial_design_budget=initial_design_budget,
        candidate_pool_size=20,
    )

    assert iterations >= 1, "No trials were evaluated"
    assert cumulative_cost <= total_resource_budget + 1e-9, (
        f"Cumulative cost {cumulative_cost:.4f} exceeded budget {total_resource_budget}"
    )
    # The manual loop uses EmptyInitialDesign, but SMAC's intensifier may
    # replay entries from the ask/tell history, so the run history has at
    # least as many entries as the iterations we explicitly counted.
    assert len(smac.runhistory) >= iterations


def test_manual_loop_transitions_from_initial_design_to_bo(
    scenario, configspace, cost_model_and_callback, smac_facade
):
    """The loop must complete at least one initial-design trial before switching to BO."""
    total_resource_budget = 2.0
    initial_design_budget = 0.5   # enough for a few manual trials

    cost_model, cost_surrogate_callback = cost_model_and_callback
    smac, acquisition_function = smac_facade

    # Track origins to confirm both phases execute
    origins_seen: list[str | None] = []

    cumulative_cost = 0.0
    candidate_pool = list(configspace.sample_configuration(size=20))
    selected_configs_manual: list[Configuration] = []

    while cumulative_cost < total_resource_budget:
        if cumulative_cost < initial_design_budget:
            if not selected_configs_manual:
                chosen_config = candidate_pool[0]
            else:
                chosen_config = candidate_pool[len(selected_configs_manual) % len(candidate_pool)]

            chosen_config.origin = "Manual Cost-Aware Initial Design"
            trial_info = TrialInfo(config=chosen_config, seed=scenario.seed)
            if chosen_config in candidate_pool:
                candidate_pool.remove(chosen_config)
            selected_configs_manual.append(chosen_config)
        else:
            if selected_configs_manual:
                selected_configs_manual = []
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
        origins_seen.append(trial_info.config.origin)
        smac.tell(
            trial_info,
            TrialValue(cost=performance, time=cost),
        )

    manual_trials = [o for o in origins_seen if o == "Manual Cost-Aware Initial Design"]
    bo_trials = [o for o in origins_seen if o != "Manual Cost-Aware Initial Design"]

    assert len(manual_trials) >= 1, "Expected at least one manual initial-design trial"
    assert len(bo_trials) >= 1, "Expected at least one BO trial after the initial design phase"


def test_manual_loop_incumbent_exists_after_run(
    scenario, configspace, cost_model_and_callback, smac_facade
):
    """After the manual loop, SMAC's intensifier should have a valid incumbent."""
    total_resource_budget = 2.0
    initial_design_budget = total_resource_budget / 4.0

    cost_model, cost_surrogate_callback = cost_model_and_callback
    smac, acquisition_function = smac_facade

    _run_manual_loop(
        smac=smac,
        acquisition_function=acquisition_function,
        cost_model=cost_model,
        configspace=configspace,
        scenario=scenario,
        total_resource_budget=total_resource_budget,
        initial_design_budget=initial_design_budget,
        candidate_pool_size=20,
    )

    incumbent = smac.intensifier.get_incumbent()
    assert incumbent is not None, "No incumbent found after optimization"

    # Incumbent must be a valid configuration within the defined bounds
    assert 0.0 <= incumbent["x"] <= 10.0, (
        f"Incumbent x={incumbent['x']:.4f} is outside [0, 10]"
    )

    # Validating the incumbent should yield finite, non-negative outputs
    result = evaluate_config(incumbent)
    assert np.isfinite(result["performance"])
    assert result["cost"] > 0.0
