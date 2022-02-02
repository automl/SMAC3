__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

from functools import partial
import unittest
import numpy as np
from matplotlib import pyplot as plt

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_mf_facade import SMAC4MF
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario


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


def plot(all_x):
    plt.figure()
    for x in all_x:
        f1, f2 = schaffer(x)
        plt.scatter(f1, f2, c="blue", alpha=0.1)

    plt.show()


def plot_from_smac(smac):
    rh = smac.get_runhistory()
    all_x = []
    for (config_id, instance_id, seed, budget), (
        cost,
        time,
        status,
        starttime,
        endtime,
        additional_info,
    ) in rh.data.items():
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
            "rng": np.random.RandomState(0),
            "tae_runner": tae,
            "multi_objective_kwargs": {"rho": 0.05},
        }

    def test_AC(self):
        smac = SMAC4AC(**self.facade_kwargs)
        incumbent = smac.optimize()

        f1_inc, f2_inc = schaffer(incumbent["x"])
        f1_opt, f2_opt = get_optimum()

        f2_inc = f2_inc / UPSCALING_FACTOR

        self.assertAlmostEqual(f1_inc + f2_inc, f1_opt + f2_opt, places=1)

        return smac


if __name__ == "__main__":
    t = SchafferTest()
    t.setUp()

    smac = t.test_AC()
    plot_from_smac(smac)
