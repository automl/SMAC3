import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac import (
    AlgorithmConfigurationFacade,
    BlackBoxFacade,
    HyperparameterOptimizationFacade,
    RandomFacade,
    Callback
)
from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.multi_objective.parego import ParEGO

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

MIN_V = -2
MAX_V = 2


def schaffer(x):
    f1 = np.square(x)
    f2 = np.square(np.sqrt(f1) - 2)

    return f1, f2


def get_optimum():
    optimum_sum = np.inf
    optimum = None

    for v in np.linspace(MIN_V, MAX_V, 200):
        f1, f2 = schaffer(v)

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


class WrapStrategy(AbstractMultiObjectiveAlgorithm):

    def __init__(
        self,
        strategy: AbstractMultiObjectiveAlgorithm,
        *args,
        **kwargs
    ):
        self.strategy = strategy(*args, **kwargs)
        self.n_calls_update_on_iteration_start = 0
        self.n_calls___call__ = 0

    def update_on_iteration_start(self) -> None:  # noqa: D102
        self.n_calls_update_on_iteration_start += 1
        return self.strategy.update_on_iteration_start()

    def __call__(self, values: list[float]) -> float:  # noqa: D102
        self.n_calls___call__ += 1
        return self.strategy(values)


@pytest.mark.parametrize(
    "facade", [BlackBoxFacade, HyperparameterOptimizationFacade, AlgorithmConfigurationFacade, RandomFacade]
)
def test_mean_aggregation(facade, make_scenario, configspace):
    scenario = make_scenario(configspace, use_multi_objective=True)

    multi_objective_algorithm = WrapStrategy(MeanAggregationStrategy, scenario=scenario)

    smac = facade(
        scenario=scenario,
        target_function=tae,
        multi_objective_algorithm=multi_objective_algorithm,
        overwrite=True,
    )
    incumbent = smac.optimize()

    f1_inc, f2_inc = schaffer(incumbent["x"])
    f1_opt, f2_opt = get_optimum()

    inc = f1_inc + f2_inc
    opt = f1_opt + f2_opt
    diff = abs(inc - opt)

    assert diff < 0.06

    assert multi_objective_algorithm.n_calls_update_on_iteration_start >= 100
    assert multi_objective_algorithm.n_calls_update_on_iteration_start <= 130
    assert multi_objective_algorithm.n_calls___call__ >= 100


@pytest.mark.parametrize(
    "facade", [BlackBoxFacade, HyperparameterOptimizationFacade, AlgorithmConfigurationFacade, RandomFacade]
)
def test_parego(facade, make_scenario, configspace):
    scenario = make_scenario(configspace, use_multi_objective=True)

    multi_objective_algorithm = WrapStrategy(ParEGO, scenario=scenario)

    smac = facade(
        scenario=scenario,
        target_function=tae,
        multi_objective_algorithm=multi_objective_algorithm,
        overwrite=True,
    )

    smac.optimize()

    # The incumbent is not ambiguous because we have a Pareto front
    confs, vals = smac.runhistory.get_pareto_front()

    min_ = np.inf
    for x, y in zip(confs, vals):
        tr = schaffer(x["x"])
        assert np.allclose(tr, y)
        if np.sum(y) < min_:
            min_ = np.sum(y)

    opt = np.sum(get_optimum())
    assert abs(np.sum(min_) - opt) <= 0.06

    assert multi_objective_algorithm.n_calls_update_on_iteration_start >= 100
    assert multi_objective_algorithm.n_calls_update_on_iteration_start <= 120
    assert multi_objective_algorithm.n_calls___call__ >= 100
