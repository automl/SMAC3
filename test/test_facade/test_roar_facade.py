import shutil
import unittest

import numpy as np

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

    def tearDown(self):
        shutil.rmtree('run_1')

    def test_inject_stats_and_runhistory_object_to_TAE(self):
        ta = ExecuteTAFuncArray(lambda x: x**2)
        self.assertIsNone(ta.stats)
        self.assertIsNone(ta.runhistory)
        ROAR(tae_runner=ta, scenario=self.scenario)
        self.assertIsInstance(ta.stats, Stats)
        self.assertIsInstance(ta.runhistory, RunHistory)