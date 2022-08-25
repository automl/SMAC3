import pytest

import numpy as np
from ConfigSpace import ConfigurationSpace, Float

from smac import AlgorithmConfigurationFacade, BlackBoxFacade, HyperparameterFacade
from smac.configspace import ConfigurationSpace
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.multi_objective.parego import ParEGO

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

MIN_V = -2
MAX_V = 2
UPSCALING_FACTOR = 2000


def schaffer(x):
    f1 = np.square(x)
    f2 = np.square(np.sqrt(f1) - 2) * UPSCALING_FACTOR

    return f1, f2


def get_optimum():
    optimum_sum = np.inf
    optimum = None

    for v in np.linspace(MIN_V, MAX_V, 200):
        f1, f2 = schaffer(v)
        f2 = f2 / UPSCALING_FACTOR

        if f1 + f2 < optimum_sum:
            optimum_sum = f1 + f2
            optimum = (f1, f2)

    return optimum


def tae(cfg):
    f1, f2 = schaffer(cfg["x"])
    return {"cost1": f1, "cost2": f2}


@pytest.fixture
def configspace():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Float("x", (MIN_V, MAX_V)))

    return cs


def test_mean_aggregation(make_scenario, configspace):
    scenario = make_scenario(configspace, use_multi_objective=True)

    for facade in [BlackBoxFacade, HyperparameterFacade, AlgorithmConfigurationFacade]:
        smac = facade(
            scenario=scenario,
            target_algorithm=tae,
            multi_objective_algorithm=MeanAggregationStrategy(scenario=scenario),
            overwrite=True,
        )
        incumbent = smac.optimize()

        f1_inc, f2_inc = schaffer(incumbent["x"])
        f1_opt, f2_opt = get_optimum()
        f2_inc = f2_inc / UPSCALING_FACTOR

        inc = f1_inc + f2_inc
        opt = f1_opt + f2_opt
        diff = abs(inc - opt)

        assert diff < 0.05


def test_parego(make_scenario, configspace):
    scenario = make_scenario(configspace, use_multi_objective=True)

    for facade in [BlackBoxFacade, HyperparameterFacade, AlgorithmConfigurationFacade]:
        smac = facade(
            scenario=scenario,
            target_algorithm=tae,
            multi_objective_algorithm=ParEGO(scenario=scenario),
            overwrite=True,
        )
        incumbent = smac.optimize()

        f1_inc, f2_inc = schaffer(incumbent["x"])
        f1_opt, f2_opt = get_optimum()
        f2_inc = f2_inc / UPSCALING_FACTOR

        inc = f1_inc + f2_inc
        opt = f1_opt + f2_opt
        diff = abs(inc - opt)

        assert diff < 0.05
