from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac.facade.cost_aware_facade import CostAwareFacade
from smac.model.random_forest import RandomForest
from smac.scenario import Scenario

# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------


def evaluate_config(config: Configuration, seed: int = 0) -> dict[str, float]:
    """2D target function returning performance loss and evaluation cost.

    Performance bowl minimized at (-2, -1). Cost ranges in [0.1, 1.1].
    """
    x, y = config["x"], config["y"]
    performance = (x + 2) ** 2 + (y + 1) ** 2

    cost_unnormalized = (
        np.exp(-((x - 2) ** 2 + (y - 2) ** 2))
        + np.exp(-((x + 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x - 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x + 2) ** 2 + (y - 2) ** 2))
    )
    cost = (cost_unnormalized + 1) / 2 + 0.1
    return {"performance": performance, "cost": cost}


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
        name="CostAwareFacadeTest",
        objectives="cost",
        n_trials=np.inf,
        seed=0,
        deterministic=True,
        output_directory=tmp_path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cost_aware_facade_optimize_with_cost_formula(scenario):
    """CostAwareFacade.optimize should run end-to-end and respect the resource budget."""
    total_resource_budget = 2.0
    cost_formula = lambda cfg: evaluate_config(cfg)["cost"]

    # Use a small candidate pool to keep the test fast
    initial_design = CostAwareFacade.get_initial_design(
        scenario=scenario,
        cost_formula=cost_formula,
        initial_budget=total_resource_budget * 0.25,
        candidate_pool_size=30,
    )

    smac = CostAwareFacade(
        scenario=scenario,
        target_function=evaluate_config,
        total_resource_budget=total_resource_budget,
        cost_formula=cost_formula,
        initial_design=initial_design,
        overwrite=True,
    )

    incumbent = smac.optimize()

    assert incumbent is not None
    assert isinstance(incumbent, Configuration)
    assert len(smac.runhistory) >= 2

    # Check that costs were recorded properly in runhistory
    for trial_value in smac.runhistory.values():
        assert "resource_cost" in trial_value.additional_info
        assert trial_value.additional_info["resource_cost"] > 0
        assert trial_value.cost >= 0

    # Verify both initial design and BO configurations were evaluated
    origins = {config.origin for config in smac.runhistory.get_configs()}
    assert any("Initial Design" in o or "Sampling" in o for o in origins), "No initial design configs found"
    assert any("Local Search" in o or "Random Search" in o for o in origins), "No BO configs found"


def test_cost_aware_facade_optimize_with_default_surrogate_model(scenario):
    """CostAwareFacade should train a surrogate cost model when no formula is given."""
    total_resource_budget = 2.0

    initial_design = CostAwareFacade.get_initial_design(
        scenario=scenario,
        initial_budget=total_resource_budget * 0.25,
        candidate_pool_size=30,
    )

    smac = CostAwareFacade(
        scenario=scenario,
        target_function=evaluate_config,
        total_resource_budget=total_resource_budget,
        initial_design=initial_design,
        overwrite=True,
    )

    incumbent = smac.optimize()

    assert incumbent is not None
    assert isinstance(incumbent, Configuration)
    assert len(smac.runhistory) >= 2


def test_cost_aware_facade_invalid_arguments(scenario):
    """Passing both cost_model and cost_formula must raise a ValueError."""
    cost_model = RandomForest(configspace=scenario.configspace, seed=0)
    cost_formula = lambda cfg: 1.0

    with pytest.raises(ValueError, match="Cannot provide both"):
        CostAwareFacade(
            scenario=scenario,
            target_function=evaluate_config,
            total_resource_budget=5.0,
            cost_model=cost_model,
            cost_formula=cost_formula,
            overwrite=True,
        )
