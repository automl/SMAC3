__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

import unittest

import numpy as np
from ConfigSpace import ConfigurationSpace, Float

from smac import (
    AlgorithmConfigurationFacade,
    BlackBoxFacade,
    HyperparameterFacade,
    Scenario,
    multi_objective,
)
from smac.configspace import ConfigurationSpace
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.multi_objective.parego import ParEGO

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


def tae(cfg):
    f1, f2 = schaffer(cfg["x"])
    return {"metric1": f1, "metric2": f2}


class SchafferTest(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameter(Float("x", (MIN_V, MAX_V)))

        # Scenario object
        self.scenario = Scenario(
            self.cs,
            n_runs=50,
            objectives=["metric1", "metric2"],
            output_directory="smac3_output_test",
        )

    def test_mean_aggregation(self):
        for facade in [BlackBoxFacade, HyperparameterFacade, AlgorithmConfigurationFacade]:
            smac = facade(
                scenario=self.scenario,
                target_algorithm=tae,
                multi_objective_algorithm=MeanAggregationStrategy(self.scenario.seed),
                overwrite=True,
            )
            incumbent = smac.optimize()

            f1_inc, f2_inc = schaffer(incumbent["x"])
            f1_opt, f2_opt = get_optimum()

            inc = f1_inc + f2_inc
            opt = f1_opt + f2_opt
            diff = abs(inc - opt)

            assert diff < 0.1

    def test_parego(self):
        for facade in [BlackBoxFacade, HyperparameterFacade, AlgorithmConfigurationFacade]:
            smac = facade(
                scenario=self.scenario,
                target_algorithm=tae,
                multi_objective_algorithm=ParEGO(seed=self.scenario.seed),
                overwrite=True,
            )
            incumbent = smac.optimize()

            f1_inc, f2_inc = schaffer(incumbent["x"])
            f1_opt, f2_opt = get_optimum()

            inc = f1_inc + f2_inc
            opt = f1_opt + f2_opt
            diff = abs(inc - opt)

            assert diff < 0.1
