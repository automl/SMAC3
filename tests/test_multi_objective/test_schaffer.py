__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

import unittest
import numpy as np
from matplotlib import pyplot as plt

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_ac_facade import SMAC4AC
from smac.optimizer.multi_objective.parego import ParEGO
from smac.scenario.scenario import Scenario


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
        plt.scatter(f1, f2, c="blue", alpha=0.1)

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
        self.cs.add_hyperparameter(
            UniformFloatHyperparameter("x", lower=MIN_V, upper=MAX_V)
        )

        # Scenario object
        self.scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternatively runtime)
                "runcount-limit": 50,  # max. number of function evaluations
                "cs": self.cs,  # configuration space
                "deterministic": True,
                "multi_objectives": "metric1, metric2",
                "limit_resources": False,
            }
        )

        self.facade_kwargs = {
            "scenario": self.scenario,
            "rng": np.random.RandomState(5),
            "tae_runner": tae,
        }

        self.parego_facade_kwargs = {
            "scenario": self.scenario,
            "rng": np.random.RandomState(5),
            "tae_runner": tae,
            "multi_objective_algorithm": ParEGO,
            "multi_objective_kwargs": {"rho": 0.05},
        }

    def test_facades(self):
        results = []
        for facade in [SMAC4BB, SMAC4HPO, SMAC4AC]:
            smac = facade(**self.facade_kwargs)
            incumbent = smac.optimize()

            f1_inc, f2_inc = schaffer(incumbent["x"])
            f1_opt, f2_opt = get_optimum()

            self.assertAlmostEqual(f1_inc + f2_inc, f1_opt + f2_opt, places=1)
            results.append(smac)

        return results


if __name__ == "__main__":
    t = SchafferTest()
    t.setUp()

    for smac in t.test_facades():
        plot_from_smac(smac)
