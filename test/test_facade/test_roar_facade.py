from contextlib import suppress
import shutil
import unittest
from nose.plugins.attrib import attr

import numpy as np

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace

from smac.runhistory.runhistory import RunHistory
from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncArray


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

    def test_inject_stats_and_runhistory_object_to_TAE(self):
        ta = ExecuteTAFuncArray(lambda x: x**2)
        self.assertIsNone(ta.stats)
        self.assertIsNone(ta.runhistory)
        ROAR(tae_runner=ta, scenario=self.scenario)
        self.assertIsInstance(ta.stats, Stats)
        self.assertIsInstance(ta.runhistory, RunHistory)

    @attr('slow')
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