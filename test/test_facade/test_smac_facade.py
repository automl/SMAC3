import unittest

from smac.configspace import ConfigurationSpace

from smac.runhistory.runhistory import RunHistory
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFunc


class TestSMACFacade(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({'cs': self.cs, 'run_obj': 'quality'})

    def test_inject_stats_and_runhistory_object_to_TAE(self):
        ta = ExecuteTAFunc(lambda x: x**2)
        self.assertIsNone(ta.stats)
        self.assertIsNone(ta.runhistory)
        SMAC(tae_runner=ta, scenario=self.scenario)
        self.assertIsInstance(ta.stats, Stats)
        self.assertIsInstance(ta.runhistory, RunHistory)
