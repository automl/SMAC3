__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

import unittest
import numpy as np

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter


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
        def tae(x):
            """x is a single continous hyperparameter."""
            f1, f2 = schaffer_n1(x)
            return {'metric1': f1, 'metric2': f2}

        # x should be evaluated in the inteval [-2, 2]
        A = 2
        n = 1000
        X = np.linspace(-A, A, n)
        true_pareto_front = schaffer_pareto(X)

        cs = ConfigurationSpace()
        UniformFloatHyperparameter('x', lower=-A, upper=A)

