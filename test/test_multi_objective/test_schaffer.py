__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

import unittest
import numpy as np

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario


def schaffer_func1(x):
    return np.square(x)


def schaffer_func2(x):
    return np.square(x - 2)


def schaffer_n1(x):
    return schaffer_func1(x), schaffer_func2(x)


def schaffer_pareto(x):
    """return estimate & true pareto front"""

    def pareto(x):
        # substituted x in f2 with x=np.sqrt(f1)
        return np.square(np.sqrt(x) - 2)

    f1 = schaffer_func1(x)
    return f1, pareto(f1)


class SchafferTest(unittest.TestCase):
    def test_Schaffer_no1(self):
        """Testing whether multi-objective function Schaffer is optimized properly using ParEGO:
        the observed values are on the pareto front."""

        def tae(cfg):
            """x is a single continuous hyperparameter.
            :param cfg: ConfigSpace object"""
            f1, f2 = schaffer_n1(cfg['x'])
            return {'metric1': f1, 'metric2': f2}

        # x should be evaluated in the inteval [-2, 2]
        A = 2
        n = 1000
        X = np.linspace(-A, A, n)
        true_pareto_front = schaffer_pareto(X)
        true_pareto_front = np.column_stack(true_pareto_front)

        cs = ConfigurationSpace()
        X = UniformFloatHyperparameter('x', lower=-A, upper=A)
        cs.add_hyperparameters([X])

        # Scenario object
        scenario = Scenario({
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount-limit": 50,  # max. number of function evaluations
            "cs": cs,  # configuration space
            "deterministic": "true",
            "multi_objectives": "metric1, metric2",
        })

        smac = SMAC4HPO(scenario=scenario,
                        rng=np.random.RandomState(42),
                        tae_runner=tae,
                        multi_objective_kwargs={
                            'rho': 0.05
                            # str or cls or callable
                        })

        smac.optimize()

        # extract the cost values from the runhistory:
        # queried_x = np.concatenate([x.get_array() for x in smac.runhistory.config_ids.keys()])
        observed_costs = np.vstack([v[0] for v in smac.runhistory.data.values()])

        # expectation on the true value of the pareto front for queried parameter configurations
        # fixme: find out why SMAC.runhistory's observed values lie on a paraboloid for observed_costs[:,1] > 4
        f1 = observed_costs[:, 0][observed_costs[:, 1] < 4]
        f2 = np.square(np.sqrt(f1) - 2)

        self.assertTrue(np.allclose(observed_costs[:, 1][observed_costs[:, 1] < 4], f2))


if __name__ == '__main__':
    unittest.main()
