from __future__ import annotations

from typing import Callable
from pathlib import Path

import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac.scenario import Scenario


def evaluate_config(config: Configuration, seed: int = 0) -> dict[str, float]:
    """2D target function returning performance loss and evaluation cost.

    Performance bowl is minimized at (-2, -1). Cost landscape has four peaks/valleys
    in [0.1, 1.0], deliberately misaligned with performance to test cost-performance
    trade-offs.
    """
    x, y = config["x"], config["y"]
    performance = (x + 2) ** 2 + (y + 1) ** 2

    cost_unnormalized = (
        np.exp(-((x - 2) ** 2 + (y - 2) ** 2))
        + np.exp(-((x + 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x - 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x + 2) ** 2 + (y - 2) ** 2))
    )
    # Normalize cost to [0.1, 1.0]
    cost = 0.1 + 0.9 * (cost_unnormalized + 1) / 2
    return {"performance": performance, "cost": cost}


@pytest.fixture
def target_function() -> Callable[[Configuration, int], dict[str, float]]:
    """Shared 2D cost-aware target function."""
    return evaluate_config


@pytest.fixture
def cost_formula(target_function: Callable[[Configuration, int], dict[str, float]]) -> Callable[[Configuration], float]:
    """Shared cost formula extracting evaluation cost from the target function."""
    return lambda cfg: target_function(cfg)["cost"]


@pytest.fixture
def configspace() -> ConfigurationSpace:
    """Standard 2D configuration space for cost-aware tests."""
    cs = ConfigurationSpace(seed=0)
    cs.add(UniformFloatHyperparameter("x", -3.5, 3.5, default_value=0))
    cs.add(UniformFloatHyperparameter("y", -3.5, 3.5, default_value=0))
    return cs


@pytest.fixture
def scenario(configspace: ConfigurationSpace, tmp_path: Path) -> Scenario:
    """Standard Scenario for cost-aware tests."""
    return Scenario(
        configspace=configspace,
        name="CostAwareTest",
        objectives="cost",
        n_trials=np.inf,
        seed=0,
        deterministic=True,
        output_directory=tmp_path,
    )
