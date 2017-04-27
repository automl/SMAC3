import unittest

import numpy as np

from smac.configspace import ConfigurationSpace

from smac.epm.base_epm import AbstractEPM
from smac.facade.smac_facade import SMAC
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.intensification.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.optimizer.acquisition import EI, AbstractAcquisitionFunction
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.tae.execute_ta_run import ExecuteTARun
from smac.utils.io.traj_logging import TrajLogger

class TestSMACFacade(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({'cs': self.cs, 'run_obj': 'quality',
                                  'output_dir': ''})

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

    def test_pass_tae_runner_objective(self):
        tae = ExecuteTAFuncDict(lambda: 1,
                                run_obj='runtime')
        self.assertRaisesRegexp(ValueError, "Objective for the target algorithm"
                                            " runner and the scenario must be "
                                            "the same, but are 'runtime' and "
                                            "'quality'",
                                SMAC, tae_runner=tae, scenario=self.scenario)

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

    def test_get_runhistory_and_trajectory_and_tae_runner(self):
        ta = ExecuteTAFuncDict(lambda x: x ** 2)
        smac = SMAC(tae_runner=ta, scenario=self.scenario)
        self.assertRaises(ValueError, smac.get_runhistory)
        self.assertRaises(ValueError, smac.get_trajectory)
        smac.trajectory = 'dummy'
        self.assertEqual(smac.get_trajectory(), 'dummy')
        smac.runhistory = 'dummy'
        self.assertEqual(smac.get_runhistory(), 'dummy')
        self.assertEqual(smac.get_tae_runner(), ta)

    def test_inject_dependencies(self):
        # initialize objects with missing dependencies
        ta = ExecuteTAFuncDict(lambda x: x ** 2)
        rh = RunHistory(aggregate_func=None)
        acqu_func = EI(model=None)
        intensifier = Intensifier(tae_runner=None,
                                  stats=None,
                                  traj_logger=None,
                                  rng=np.random.RandomState(),
                                  instances=None)
        init_design = DefaultConfiguration(tae_runner=None,
                                           scenario=None,
                                           stats=None,
                                           traj_logger=None,
                                           rng=np.random.RandomState())
        rh2epm = RunHistory2EPM4Cost(scenario=self.scenario, num_params=0)
        rh2epm.scenario = None

        # assert missing dependencies
        self.assertIsNone(rh.aggregate_func)
        self.assertIsNone(acqu_func.model)
        self.assertIsNone(intensifier.tae_runner)
        self.assertIsNone(intensifier.stats)
        self.assertIsNone(intensifier.traj_logger)
        self.assertIsNone(init_design.tae_runner)
        self.assertIsNone(init_design.scenario)
        self.assertIsNone(init_design.stats)
        self.assertIsNone(init_design.traj_logger)
        self.assertIsNone(rh2epm.scenario)

        # initialize smac-object
        SMAC(scenario=self.scenario,
             tae_runner=ta,
             runhistory=rh,
             intensifier=intensifier,
             acquisition_function=acqu_func,
             runhistory2epm=rh2epm,
             initial_design=init_design)

        # assert that missing dependencies are injected
        self.assertIsNotNone(rh.aggregate_func, AbstractAcquisitionFunction)
        self.assertIsInstance(acqu_func.model, AbstractEPM)
        self.assertIsInstance(intensifier.tae_runner, ExecuteTARun)
        self.assertIsInstance(intensifier.stats, Stats)
        self.assertIsInstance(intensifier.traj_logger, TrajLogger)
        self.assertIsInstance(init_design.tae_runner, ExecuteTARun)
        self.assertIsInstance(init_design.scenario, Scenario)
        self.assertIsInstance(init_design.stats, Stats)
        self.assertIsInstance(init_design.traj_logger, TrajLogger)
        self.assertIsInstance(rh2epm.scenario, Scenario)
