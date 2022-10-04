import logging
import warnings

import pytest
from ConfigSpace import Configuration

from smac import HyperparameterOptimizationFacade
from smac.intensifier import SuccessiveHalving


def test_success(make_scenario, configspace_small):
    scenario = make_scenario(configspace_small)

    # ---------------------------------------------------------------------------------
    # Fail because no seed

    def tf(config: Configuration) -> float:
        return 0

    with pytest.raises(RuntimeError, match="Target function needs .* seed"):
        HyperparameterOptimizationFacade(scenario, tf, overwrite=True)

    # ---------------------------------------------------------------------------------
    # That should work now

    def tf(config: Configuration, seed: int) -> float:
        return 0

    HyperparameterOptimizationFacade(scenario, tf, overwrite=True)

    # ---------------------------------------------------------------------------------
    # Now we need budget too

    def tf(config: Configuration, seed: int) -> float:
        return 0

    # Change intensifier
    intensifier = SuccessiveHalving(scenario)
    with pytest.raises(RuntimeError, match="Target function needs .* budget"):
        HyperparameterOptimizationFacade(scenario, tf, intensifier=intensifier, overwrite=True)

    # ---------------------------------------------------------------------------------
    # Add budget

    def tf(config: Configuration, seed: int, budget: float) -> float:
        return 0

    # Change intensifier
    intensifier = SuccessiveHalving(scenario)
    HyperparameterOptimizationFacade(scenario, tf, intensifier=intensifier, overwrite=True)

    # ---------------------------------------------------------------------------------
    # Now we use instances (we only can have budget or instance).
    # Since we only specified budget, we should get into trouble here.

    def tf(config: Configuration, seed: int, budget: float) -> float:
        return 0

    # Change intensifier
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=100)
    intensifier = SuccessiveHalving(scenario)
    with pytest.raises(RuntimeError, match="Target function needs .* instance"):
        HyperparameterOptimizationFacade(scenario, tf, intensifier=intensifier, overwrite=True)

    # ---------------------------------------------------------------------------------
    # Add instance as argument (we only can have budget or instance)

    def tf(config: Configuration, seed: int, instance: float) -> float:
        return 0

    # Change intensifier
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=100)
    intensifier = SuccessiveHalving(scenario)
    HyperparameterOptimizationFacade(scenario, tf, intensifier=intensifier, overwrite=True)
