from contextlib import suppress
import shutil
import unittest

import numpy as np

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace

from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestROARFacade(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({'cs': self.cs, 'run_obj': 'quality',
                                  'output_dir': ''})
        self.output_dirs = []

    def tearDown(self):
        shutil.rmtree('run_1', ignore_errors=True)
        for i in range(20):
            with suppress(Exception):
                dirname = 'run_1' + ('.OLD' * i)
                shutil.rmtree(dirname)
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)

    def test_check_deterministic_rosenbrock(self):
        def rosenbrock_2d(x):
            x1 = x['x1']
            x2 = x['x2']
            val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
            return val

        def opt_rosenbrock():
            cs = ConfigurationSpace()

            cs.add_hyperparameter(UniformFloatHyperparameter("x1", -5, 5, default_value=-3))
            cs.add_hyperparameter(UniformFloatHyperparameter("x2", -5, 5, default_value=-4))

            scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                                 "runcount-limit": 50,  # maximum function evaluations
                                 "cs": cs,  # configuration space
                                 "deterministic": "true",
                                 "intensification_percentage": 0.000000001
                                 })

            roar = ROAR(scenario=scenario, rng=np.random.RandomState(42),
                        tae_runner=rosenbrock_2d)
            incumbent = roar.optimize()
            return incumbent, roar.scenario.output_dir

        i1, output_dir = opt_rosenbrock()
        self.output_dirs.append(output_dir)
        x1_1 = i1.get('x1')
        x2_1 = i1.get('x2')
        i2, output_dir = opt_rosenbrock()
        self.output_dirs.append(output_dir)
        x1_2 = i2.get('x1')
        x2_2 = i2.get('x2')
        self.assertAlmostEqual(x1_1, x1_2)
        self.assertAlmostEqual(x2_1, x2_2)
