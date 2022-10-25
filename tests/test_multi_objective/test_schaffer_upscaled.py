import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac import (
    AlgorithmConfigurationFacade,
    BlackBoxFacade,
    HyperparameterOptimizationFacade,
)
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


def tae(cfg, seed=0):
    f1, f2 = schaffer(cfg["x"])
    return {"cost1": f1, "cost2": f2}


@pytest.fixture
def configspace():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Float("x", (MIN_V, MAX_V)))

    return cs


@pytest.mark.parametrize("facade", [BlackBoxFacade, HyperparameterOptimizationFacade, AlgorithmConfigurationFacade])
def test_mean_aggregation(facade, make_scenario, configspace):
    scenario = make_scenario(configspace, use_multi_objective=True)

    smac = facade(
        scenario=scenario,
        target_function=tae,
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

    assert diff < 0.06


@pytest.mark.parametrize("facade", [BlackBoxFacade, HyperparameterOptimizationFacade, AlgorithmConfigurationFacade])
def test_parego(facade, make_scenario, configspace):
    scenario = make_scenario(configspace, use_multi_objective=True)

    smac = facade(
        scenario=scenario,
        target_function=tae,
        multi_objective_algorithm=ParEGO(scenario=scenario),
        overwrite=True,
    )
    smac.optimize()

    # The incumbent is not ambiguous because we have a Pareto front
    confs, vals = smac.runhistory.get_pareto_front()

    min_ = [np.inf, np.inf]
    for x, y in zip(confs, vals):
        tr = schaffer(x["x"])
        assert np.allclose(tr, y)
        if y[0] + y[1] / UPSCALING_FACTOR < min_[0] + min_[1] / UPSCALING_FACTOR:
            min_ = y

    opt = np.sum(get_optimum())
    f1_inc, f2_inc = min_
    f2_inc = f2_inc / UPSCALING_FACTOR
    min_ = (f1_inc, f2_inc)
    assert abs(np.sum(min_) - opt) <= 0.06
