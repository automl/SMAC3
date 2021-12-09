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
    f1 = schaffer_func1(x)
    pareto = lambda x: np.square(np.sqrt(x) - 2)  # substituted x in f2 with x=np.sqrt(f1)
    return f1, pareto(f1)


class MultiObjectiveTest(unittest.TestCase):
    def test_Schaffer_no1(self):
        """Testing whether multi-objective function Schaffer is optimized properly using
        ParEGO"""
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

        cs = ConfigurationSpace()
        X = UniformFloatHyperparameter('x', lower=-A, upper=A)
        cs.add_hyperparameters([X])

        # check config space & tae:
        # tae(cs.get_default_configuration())

        # Scenario object
        scenario = Scenario({
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount-limit": 50,  # max. number of function evaluations
            "cs": cs,  # configuration space
            "deterministic": "true"})

        smac = SMAC4HPO(scenario=scenario,
                        rng=np.random.RandomState(42),
                        tae_runner=tae,
                        multi_objective_kwargs={
                            'rho': 0.05,
                            'algorithm': 'par_ego'  # str or cls or callable
                        })

        incumbent = smac.optimize()

        # TODO: find out the exact format in which smac returns the multi-objective
        #  values. - then check reasonable convergence bounds & how to properly set up the test.
        # TODO add the
        smac.runhistory.multi_objective
        self.assertAlmostEqual(None, true_pareto_front(None))


if __name__ == '__main__':
    unittest.main()
