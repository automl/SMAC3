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
        self.scenario = Scenario({'cs': self.cs, 'run_obj': 'quality'})

    def test_inject_stats_and_runhistory_object_to_TAE(self):
        ta = ExecuteTAFuncArray(lambda x: x**2)
        self.assertIsNone(ta.stats)
        self.assertIsNone(ta.runhistory)
        ROAR(tae_runner=ta, scenario=self.scenario)
        self.assertIsInstance(ta.stats, Stats)
        self.assertIsInstance(ta.runhistory, RunHistory)

    def test_check_random_states(self):
        ta = ExecuteTAFuncArray(lambda x: x**2)
        # Get state immediately or it will change with the next call

        # Check whether different seeds give different random states
        S1 = ROAR(tae_runner=ta, scenario=self.scenario, rng=1)
        S1 = S1.solver.scenario.cs.random

        S2 = ROAR(tae_runner=ta, scenario=self.scenario, rng=2)
        S2 = S2.solver.scenario.cs.random
        self.assertNotEqual(sum(S1.get_state()[1] - S2.get_state()[1]), 0)

        # Check whether no seeds give different random states
        S1 = ROAR(tae_runner=ta, scenario=self.scenario)
        S1 = S1.solver.scenario.cs.random

        S2 = ROAR(tae_runner=ta, scenario=self.scenario)
        S2 = S2.solver.scenario.cs.random
        self.assertNotEqual(sum(S1.get_state()[1] - S2.get_state()[1]), 0)

        # Check whether the same seeds give the same random states
        S1 = ROAR(tae_runner=ta, scenario=self.scenario, rng=1)
        S1 = S1.solver.scenario.cs.random

        S2 = ROAR(tae_runner=ta, scenario=self.scenario, rng=1)
        S2 = S2.solver.scenario.cs.random
        self.assertEqual(sum(S1.get_state()[1] - S2.get_state()[1]), 0)

        # Check whether the same RandomStates give the same random states
        S1 = ROAR(tae_runner=ta, scenario=self.scenario,
                  rng=np.random.RandomState(1))
        S1 = S1.solver.scenario.cs.random

        S2 = ROAR(tae_runner=ta, scenario=self.scenario,
                  rng=np.random.RandomState(1))
        S2 = S2.solver.scenario.cs.random
        self.assertEqual(sum(S1.get_state()[1] - S2.get_state()[1]), 0)
