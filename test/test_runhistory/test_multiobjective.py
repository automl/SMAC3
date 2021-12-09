__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

import unittest

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter


class MultiObjectiveTest(unittest.TestCase):
    def test_Schaffer_no1(self):
        def tae(x):
            """x is a single continous hyperparameter."""
            return {'metric1': x ** 2, 'metric2': (x - 2) ** 2}

        cs = ConfigurationSpace()
        UniformFloatHyperparameter('x', lower=0., upper=1.)
