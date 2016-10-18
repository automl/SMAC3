import unittest

from smac.configspace import ConfigurationSpace

from smac.runhistory.runhistory import RunHistory
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict


class TestSMACFacade(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({'cs': self.cs, 'run_obj': 'quality'})

    def test_inject_stats_and_runhistory_object_to_TAE(self):
        ta = ExecuteTAFuncDict(lambda x: x**2)
        self.assertIsNone(ta.stats)
        self.assertIsNone(ta.runhistory)
        SMAC(tae_runner=ta, scenario=self.scenario)
        self.assertIsInstance(ta.stats, Stats)
        self.assertIsInstance(ta.runhistory, RunHistory)

    def test_pass_callable(self):
        # Check that SMAC accepts a callable as target algorithm and that it is
        # correctly wrapped with ExecuteTaFunc
        def target_algorithm(conf, inst):
            return 5
        smac = SMAC(tae_runner=target_algorithm, scenario=self.scenario)
        self.assertIsInstance(smac.solver.intensifier.tae_runner,
                              ExecuteTAFuncDict)
        self.assertIs(smac.solver.intensifier.tae_runner.ta, target_algorithm)

    def test_pass_invalid_tae_runner(self):
        self.assertRaisesRegexp(TypeError, "Argument 'tae_runner' is <class "
                                           "'int'>, but must be either a "
                                           "callable or an instance of "
                                           "ExecuteTaRun.",
                                SMAC, tae_runner=1, scenario=self.scenario)
