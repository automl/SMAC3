from __future__ import annotations

import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac import HPIFacade, Scenario
from smac.acquisition.maximizer import HPIRandomSearch
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.multi_objective.parego import ParEGO

pytest.importorskip("hypershap")


def _scenario(objectives):
    cs = ConfigurationSpace(seed=0)
    cs.add([Float("a", (0, 1)), Float("b", (0, 1))])
    return Scenario(cs, objectives=objectives, n_trials=20, deterministic=True)


def test_components():
    scenario = _scenario(["x", "y"])

    assert isinstance(HPIFacade.get_acquisition_maximizer(scenario), HPIRandomSearch)
    assert isinstance(HPIFacade.get_multi_objective_algorithm(scenario), ParEGO)
    assert HPIFacade.get_multi_objective_algorithm(scenario)._reweigh == 10
    assert HPIFacade.get_config_selector(scenario)._retrain_after == 2


def test_single_objective_uses_mean_aggregation():
    assert isinstance(HPIFacade.get_multi_objective_algorithm(_scenario("x")), MeanAggregationStrategy)


def test_optimize_multi_objective():
    scenario = _scenario(["x", "y"])
    smac = HPIFacade(scenario, lambda config, seed=0: {"x": config["a"], "y": config["b"]}, overwrite=True)

    assert len(smac.optimize()) > 0
