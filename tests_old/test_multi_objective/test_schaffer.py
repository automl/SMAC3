__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

import unittest

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from matplotlib import pyplot as plt

from smac.cli.scenario import Scenario
from smac.configspace import ConfigurationSpace
from smac.facade.algorithm_configuration_facade import AlgorithmConfigurationFacade
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.facade.hyperparameter_facade import SMAC4HPO
from smac.facade.random_facade import ROAR
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


def plot(all_x):
    plt.figure()

    for x in all_x:
        f1, f2 = schaffer(x)
        plt.scatter(f1, f2, c="blue", alpha=0.2, zorder=3000)

    plt.vlines([1], 0, 4, linestyles="dashed", colors=["red"])
    plt.hlines([1], 0, 4, linestyles="dashed", colors=["red"])

    plt.show()


def plot_from_smac(smac):
    rh = smac.get_runhistory()
    all_x = []
    for (config_id, _, _, _) in rh.data.keys():
        config = rh.ids_config[config_id]
        all_x.append(config["x"])

    plot(all_x)


def tae(cfg):
    f1, f2 = schaffer(cfg["x"])
    return {"metric1": f1, "metric2": f2}


class SchafferTest(unittest.TestCase):
    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameter(UniformFloatHyperparameter("x", lower=MIN_V, upper=MAX_V))

        # Scenario object
        self.scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternatively runtime)
                "runcount-limit": 25,  # max. number of function evaluations
                "cs": self.cs,  # configuration space
                "deterministic": True,
                "multi_objectives": "metric1, metric2",
                "limit_resources": False,
            }
        )

        self.facade_kwargs = {
            "scenario": self.scenario,
            "rng": np.random.RandomState(0),
            "tae_runner": tae,
        }

        self.parego_facade_kwargs = {
            "scenario": self.scenario,
            "rng": np.random.RandomState(0),
            "tae_runner": tae,
            "multi_objective_algorithm": ParEGO,
            "multi_objective_kwargs": {"rho": 0.05},
        }

    def test_facades(self):
        results = []
        for facade in [ROAR, BlackBoxFacade, SMAC4HPO, AlgorithmConfigurationFacade]:
            for kwargs in [self.facade_kwargs, self.parego_facade_kwargs]:
                smac = facade(**kwargs)
                incumbent = smac.optimize()

                f1_inc, f2_inc = schaffer(incumbent["x"])
                f1_opt, f2_opt = get_optimum()

                inc = f1_inc + f2_inc
                opt = f1_opt + f2_opt
                diff = abs(inc - opt)

                assert diff < 0.5
                results.append(smac)

        return results


if __name__ == "__main__":
    t = SchafferTest()
    t.setUp()

    for smac in t.test_facades():
        plot_from_smac(smac)
