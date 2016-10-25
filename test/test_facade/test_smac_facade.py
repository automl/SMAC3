import unittest

import numpy as np

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

    def test_check_random_states(self):
        ta = ExecuteTAFuncDict(lambda x: x**2)

        # Get state immediately or it will change with the next call

        # Check whether different seeds give different random states
        S1 = SMAC(tae_runner=ta, scenario=self.scenario, rng=1)
        S1 = S1.solver.scenario.cs.random

        S2 = SMAC(tae_runner=ta, scenario=self.scenario, rng=2)
        S2 = S2.solver.scenario.cs.random
        self.assertNotEqual(sum(S1.get_state()[1] - S2.get_state()[1]), 0)

        # Check whether no seeds give different random states
        S1 = SMAC(tae_runner=ta, scenario=self.scenario)
        S1 = S1.solver.scenario.cs.random

        S2 = SMAC(tae_runner=ta, scenario=self.scenario)
        S2 = S2.solver.scenario.cs.random
        self.assertNotEqual(sum(S1.get_state()[1] - S2.get_state()[1]), 0)

        # Check whether the same seeds give the same random states
        S1 = SMAC(tae_runner=ta, scenario=self.scenario, rng=1)
        S1 = S1.solver.scenario.cs.random

        S2 = SMAC(tae_runner=ta, scenario=self.scenario, rng=1)
        S2 = S2.solver.scenario.cs.random
        self.assertEqual(sum(S1.get_state()[1] - S2.get_state()[1]), 0)

        # Check whether the same RandomStates give the same random states
        S1 = SMAC(tae_runner=ta, scenario=self.scenario,
                  rng=np.random.RandomState(1))
        S1 = S1.solver.scenario.cs.random

        S2 = SMAC(tae_runner=ta, scenario=self.scenario,
                  rng=np.random.RandomState(1))
        S2 = S2.solver.scenario.cs.random
        self.assertEqual(sum(S1.get_state()[1] - S2.get_state()[1]), 0)
