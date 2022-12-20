from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac import AlgorithmConfigurationFacade as ACFacade
from smac import BlackBoxFacade as BBFacade
from smac import HyperbandFacade as HBFacade
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import MultiFidelityFacade as MFFacade
from smac import RandomFacade as RFacade
from smac import Scenario
from smac.initial_design.random_design import RandomInitialDesign
from smac.intensifier.intensifier import Intensifier
from smac.main.config_selector import ConfigSelector
from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.multi_objective.parego import ParEGO
from smac.scenario import Scenario

FACADES = [BBFacade, HPOFacade, MFFacade, RFacade, HBFacade, ACFacade]

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


MIN_X = -4
MAX_X = 4


def func(x):
    """Pareto points should not be between -2 and 2."""
    if x <= -2:
        y = -x
    elif x >= 2:
        y = -x + 4
    else:
        y = -(x * x) + 6

    return x, y


def tae(cfg, seed=0):
    f1, f2 = func(cfg["x"])
    return {"cost1": f1, "cost2": f2}


@pytest.fixture
def configspace():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Float("x", (MIN_X, MAX_X), default=MIN_X))

    return cs


class WrapStrategy(AbstractMultiObjectiveAlgorithm):
    def __init__(self, strategy: AbstractMultiObjectiveAlgorithm, *args, **kwargs):
        self._strategy = strategy(*args, **kwargs)
        self._n_calls_update_on_iteration_start = 0

    def update_on_iteration_start(self) -> None:  # noqa: D102
        self._n_calls_update_on_iteration_start += 1
        return self._strategy.update_on_iteration_start()

    def __call__(self, values: list[float]) -> float:  # noqa: D102
        return self._strategy(values)


@pytest.mark.parametrize("facade", FACADES)
def test_mean_aggregation(facade, make_scenario, configspace):
    """Tests whether mean aggregation works."""
    N_TRIALS = 64
    RETRAIN_AFTER = 8

    scenario: Scenario = make_scenario(configspace, use_multi_objective=True, n_trials=N_TRIALS)
    # TODO: Check whether different weighting affects the sampled configurations.
    multi_objective_algorithm = WrapStrategy(MeanAggregationStrategy, scenario=scenario)
    intensifier = Intensifier(scenario, max_config_calls=1, max_incumbents=10)
    config_selector = ConfigSelector(scenario, retrain_after=RETRAIN_AFTER)
    initial_design = RandomInitialDesign(scenario, n_configs=1)

    smac = facade(
        scenario=scenario,
        target_function=tae,
        multi_objective_algorithm=multi_objective_algorithm,
        intensifier=intensifier,
        config_selector=config_selector,
        initial_design=initial_design,
        overwrite=True,
    )
    incumbents = smac.optimize()

    # We sort the incumbents by their x values and then make sure that the current y is
    # smaller than the previous one.
    sorted_incumbents = []
    for incumbent in incumbents:
        x, y = func(incumbent["x"])
        sorted_incumbents.append((x, y))

    sorted_incumbents = sorted(sorted_incumbents, key=lambda x: x[0])
    previous_y = np.inf
    for x, y in sorted_incumbents:
        assert y <= previous_y
        previous_y = y

    # We expect N_TRIALS/RETRAIN_AFTER updates
    assert multi_objective_algorithm._n_calls_update_on_iteration_start == int(N_TRIALS / RETRAIN_AFTER)


@pytest.mark.parametrize("facade", FACADES)
def test_parego(facade, make_scenario, configspace):
    """Tests whether ParEGO works."""
    N_TRIALS = 64
    RETRAIN_AFTER = 8

    scenario: Scenario = make_scenario(configspace, use_multi_objective=True, n_trials=N_TRIALS)
    multi_objective_algorithm = WrapStrategy(ParEGO, scenario=scenario)
    intensifier = Intensifier(scenario, max_config_calls=1, max_incumbents=10)
    config_selector = ConfigSelector(scenario, retrain_after=RETRAIN_AFTER)
    initial_design = RandomInitialDesign(scenario, n_configs=1)

    smac = facade(
        scenario=scenario,
        target_function=tae,
        multi_objective_algorithm=multi_objective_algorithm,
        intensifier=intensifier,
        config_selector=config_selector,
        initial_design=initial_design,
        overwrite=True,
    )
    incumbents = smac.optimize()

    sorted_incumbents = []
    for incumbent in incumbents:
        x, y = func(incumbent["x"])
        sorted_incumbents.append((x, y))

    sorted_incumbents = sorted(sorted_incumbents, key=lambda x: x[0])
    previous_y = np.inf
    for x, y in sorted_incumbents:
        assert y <= previous_y
        previous_y = y

    # We expect N_TRIALS/RETRAIN_AFTER updates
    assert multi_objective_algorithm._n_calls_update_on_iteration_start == int(N_TRIALS / RETRAIN_AFTER)
