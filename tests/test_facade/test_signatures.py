import logging
import warnings

import pytest
from ConfigSpace import Configuration

from smac import HyperparameterFacade
from smac.intensification import SuccessiveHalving


def test_success(make_scenario, configspace_small):
    scenario = make_scenario(configspace_small)

    # ---------------------------------------------------------------------------------
    # Fail because no seed

    def tf(config: Configuration) -> float:
        return 0

    with pytest.raises(RuntimeError, match="Target function needs .* seed"):
        HyperparameterFacade(scenario, tf)

    # ---------------------------------------------------------------------------------
    # That should work now

    def tf(config: Configuration, seed: int) -> float:
        return 0

    HyperparameterFacade(scenario, tf)

    # ---------------------------------------------------------------------------------
    # Now we need budget too

    def tf(config: Configuration, seed: int) -> float:
        return 0

    # Change intensifier
    intensifier = SuccessiveHalving(scenario)
    with pytest.raises(RuntimeError, match="Target function needs .* budget"):
        HyperparameterFacade(scenario, tf, intensifier=intensifier)

    # ---------------------------------------------------------------------------------
    # Add budget

    def tf(config: Configuration, seed: int, budget: float) -> float:
        return 0

    # Change intensifier
    intensifier = SuccessiveHalving(scenario)
    HyperparameterFacade(scenario, tf, intensifier=intensifier)

    # ---------------------------------------------------------------------------------
    # Now we use instances (we only can have budget or instance).
    # Since we only specified budget, we should get into trouble here.

    def tf(config: Configuration, seed: int, budget: float) -> float:
        return 0

    # Change intensifier
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=100)
    intensifier = SuccessiveHalving(scenario)
    with pytest.raises(RuntimeError, match="Target function needs .* instance"):
        HyperparameterFacade(scenario, tf, intensifier=intensifier)

    # ---------------------------------------------------------------------------------
    # Add instance as argument (we only can have budget or instance)

    def tf(config: Configuration, seed: int, instance: float) -> float:
        return 0

    # Change intensifier
    scenario = make_scenario(configspace_small, use_instances=True, n_instances=100)
    intensifier = SuccessiveHalving(scenario)
    HyperparameterFacade(scenario, tf, intensifier=intensifier)
