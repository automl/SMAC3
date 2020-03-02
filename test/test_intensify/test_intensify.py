import collections
import unittest

import logging
import numpy as np
import time

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.intensification.intensification import Intensifier, IntensifierStage
from smac.facade.smac_ac_facade import SMAC4AC
from smac.tae.execute_ta_run import StatusType
from smac.utils.io.traj_logging import TrajLogger


def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='a',
                                                       lower=0,
                                                       upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='b',
                                                       lower=0,
                                                       upper=100))
    return cs


class TestIntensify(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.rh = RunHistory()
        self.cs = get_config_space()
        self.config1 = Configuration(self.cs,
                                     values={'a': 0, 'b': 100})
        self.config2 = Configuration(self.cs,
                                     values={'a': 100, 'b': 0})
        self.config3 = Configuration(self.cs,
                                     values={'a': 100, 'b': 100})

        self.scen = Scenario({"cutoff_time": 2, 'cs': self.cs,
                              "run_obj": 'runtime',
                              "output_dir": ''})
        self.stats = Stats(scenario=self.scen)
        self.stats.start_timing()

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

    def test_race_challenger(self):
        """
           test _race_challenger without adaptive capping
        """

        def target(x):
            return (x['a'] + 1) / 1000.
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1], run_obj_time=False)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)
        intensifier.N = 1

        inc = intensifier._race_challenger(challenger=self.config2,
                                           incumbent=self.config1,
                                           run_history=self.rh)

        self.assertEqual(inc, self.config2)

    def test_race_challenger_2(self):
        """
           test _race_challenger with adaptive capping
        """

        def target(x):
            time.sleep(1.5)
            return (x['a'] + 1) / 1000.
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime")
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1])

        self.rh.add(config=self.config1, cost=.001, time=0.001,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=12345,
                    additional_info=None)
        intensifier.N = 1

        # config2 should have a timeout (due to adaptive capping)
        # and config1 should still be the incumbent
        inc = intensifier._race_challenger(challenger=self.config2,
                                           incumbent=self.config1,
                                           run_history=self.rh,)

        # self.assertTrue(False)
        self.assertEqual(inc, self.config1)

    def test_race_challenger_3(self):
        """
           test _race_challenger with adaptive capping on a previously capped configuration
        """

        def target(config: Configuration, seed: int, instance: str):
            if instance == 1:
                time.sleep(2.1)
            else:
                time.sleep(0.6)
            return (config['a'] + 1) / 1000.
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime", par_factor=1)
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            cutoff=2,
            instances=[1])

        self.rh.add(config=self.config1, cost=0.5, time=.5,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=12345,
                    additional_info=None)

        # config2 should have a timeout (due to adaptive capping)
        # and config1 should still be the incumbent
        config, _ = intensifier.get_next_challenger(challengers=[self.config2, self.config3], chooser=None)
        inc = intensifier._race_challenger(challenger=config,
                                           incumbent=self.config1,
                                           run_history=self.rh,)
        self.assertEqual(inc, self.config1)

        # further run for incumbent
        self.rh.add(config=self.config1, cost=2, time=2,
                    status=StatusType.TIMEOUT, instance_id=2,
                    seed=12345,
                    additional_info=None)

        # give config2 a second chance - now it should run on both instances

        # run on instance 1
        config, _ = intensifier.get_next_challenger(challengers=[self.config2, self.config3], chooser=None)
        inc = intensifier._race_challenger(challenger=config,
                                           incumbent=self.config1,
                                           run_history=self.rh,)

        # run on instance 2
        config, _ = intensifier.get_next_challenger(challengers=[self.config3], chooser=None)
        self.assertEqual(config, self.config2)
        self.assertTrue(intensifier.continue_challenger)

        inc = intensifier._race_challenger(challenger=config,
                                           incumbent=self.config1,
                                           run_history=self.rh,)

        # the incumbent should still be config1 because
        # config2 should get on inst 1 a full timeout
        # such that c(config1) = 1.25 and c(config2) close to 1.3
        self.assertEqual(inc, self.config1)
        # the capped run should not be counted in runs_perf_config
        self.assertAlmostEqual(self.rh.num_runs_per_config[2], 2)
        self.assertFalse(intensifier.continue_challenger)

    def test_race_challenger_large(self):
        """
           test _race_challenger using solution_quality
        """

        def target(x):
            return 1

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=list(range(10)), run_obj_time=False,
            deterministic=True)

        for i in range(10):
            self.rh.add(config=self.config1, cost=i + 1, time=1,
                        status=StatusType.SUCCESS, instance_id=i,
                        seed=12345,
                        additional_info=None)

        intensifier.stage = IntensifierStage.RUN_CHALLENGER

        # tie on first instances and then challenger should always win
        # and be returned as inc
        while True:
            config, _ = intensifier.get_next_challenger(challengers=[self.config2, self.config3], chooser=None)
            inc = intensifier._race_challenger(challenger=config,
                                               incumbent=self.config1,
                                               run_history=self.rh,)

            # stop when challenger evaluation is over
            if not intensifier.stage == IntensifierStage.RUN_CHALLENGER:
                break

        self.assertEqual(inc, self.config2)
        self.assertEqual(self.rh.get_cost(self.config2), 1)

        # get data for config2 to check that the correct run was performed
        runs = self.rh.get_runs_for_config(self.config2, only_max_observed_budget=True)
        self.assertEqual(len(runs), 10)

    def test_race_challenger_large_blocked_seed(self):
        """
           test _race_challenger whether seeds are blocked for challenger runs
        """

        def target(x):
            return 1

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=list(range(10)), run_obj_time=False,
            deterministic=False)

        for i in range(10):
            self.rh.add(config=self.config1, cost=i + 1, time=1,
                        status=StatusType.SUCCESS, instance_id=i,
                        seed=i,
                        additional_info=None)

        intensifier.stage = IntensifierStage.RUN_CHALLENGER

        # tie on first instances and then challenger should always win
        # and be returned as inc
        while True:
            config, _ = intensifier.get_next_challenger(challengers=[self.config2, self.config3], chooser=None)
            inc = intensifier._race_challenger(challenger=config,
                                               incumbent=self.config1,
                                               run_history=self.rh,)

            # stop when challenger evaluation is over
            if not intensifier.stage == IntensifierStage.RUN_CHALLENGER:
                break

        self.assertEqual(inc, self.config2)
        self.assertEqual(self.rh.get_cost(self.config2), 1)

        # get data for config2 to check that the correct run was performed
        runs = self.rh.get_runs_for_config(self.config2, only_max_observed_budget=True)
        self.assertEqual(len(runs), 10)

        seeds = sorted([r.seed for r in runs])
        self.assertEqual(seeds, list(range(10)), seeds)

    def test_add_inc_run_det(self):
        """
            test _add_inc_run()
        """

        def target(x):
            return (x['a'] + 1) / 1000.
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="solution_quality")
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1],
            deterministic=True)

        intensifier._add_inc_run(incumbent=self.config1, run_history=self.rh)
        self.assertEqual(len(self.rh.data), 1, self.rh.data)

        # since we assume deterministic=1,
        # the second call should not add any more runs
        # given only one instance
        intensifier._add_inc_run(incumbent=self.config1, run_history=self.rh)
        self.assertEqual(len(self.rh.data), 1, self.rh.data)

    def test_add_inc_run_nondet(self):
        """
            test _add_inc_run()
        """

        def target(x):
            return (x['a'] + 1) / 1000.
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, runhistory=self.rh, run_obj="solution_quality")

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1, 2],
            deterministic=False)

        intensifier._add_inc_run(incumbent=self.config1, run_history=self.rh)
        self.assertEqual(len(self.rh.data), 1, self.rh.data)

        intensifier._add_inc_run(incumbent=self.config1, run_history=self.rh)
        self.assertEqual(len(self.rh.data), 2, self.rh.data)
        runs = self.rh.get_runs_for_config(config=self.config1, only_max_observed_budget=True)
        # exactly one run on each instance
        self.assertIn(1, [runs[0].instance, runs[1].instance])
        self.assertIn(2, [runs[0].instance, runs[1].instance])

        intensifier._add_inc_run(incumbent=self.config1, run_history=self.rh)
        self.assertEqual(len(self.rh.data), 3, self.rh.data)

    def test_get_next_challenger(self):
        """
            test get_next_challenger()
        """
        intensifier = Intensifier(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1],
            deterministic=True)

        intensifier.stage = IntensifierStage.RUN_CHALLENGER

        # get a new challenger to evaluate
        config, new = intensifier.get_next_challenger(challengers=[self.config1, self.config2], chooser=None)

        self.assertEqual(config, self.config1, intensifier.current_challenger)
        self.assertEqual(intensifier._chall_indx, 1)
        self.assertEqual(intensifier.N, 1)
        self.assertTrue(new)

        # when already evaluating a challenger, return the same challenger
        intensifier.to_run = [(1, 1, 0)]
        config, new = intensifier.get_next_challenger(challengers=[self.config2], chooser=None)
        self.assertEqual(config, self.config1, intensifier.current_challenger)
        self.assertEqual(intensifier._chall_indx, 1)
        self.assertFalse(new)

    def test_generate_challenger(self):
        """
            test generate_challenger()
        """
        # test get generator from a list of challengers
        intensifier = Intensifier(
            tae_runner=None, stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), instances=[1],
            deterministic=True)

        gen = intensifier._generate_challengers(challengers=[self.config1, self.config2], chooser=None)

        self.assertEqual(next(gen), self.config1)
        self.assertEqual(next(gen), self.config2)
        self.assertRaises(StopIteration, next, gen)

        # test get generator from a chooser - would return only 1 configuration
        intensifier = Intensifier(
            tae_runner=None, stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), instances=[1],
            deterministic=True)
        chooser = SMAC4AC(self.scen, rng=1).solver.epm_chooser

        gen = intensifier._generate_challengers(challengers=None, chooser=chooser)

        self.assertEqual(next(gen).get_dictionary(), {'a': 24, 'b': 68})
        self.assertRaises(StopIteration, next, gen)

        # when both are none, raise error
        with self.assertRaisesRegex(ValueError, "No configurations/chooser provided"):
            intensifier._generate_challengers(challengers=None, chooser=None)

    def test_eval_challenger(self):
        """
            test eval_challenger() - a complete intensification run
        """
        def target(x):
            return x['a']

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1], run_obj_time=False,
            deterministic=False, always_race_against=self.config3, run_limit=1)

        # run incumbent first if it was not run before
        config, _ = intensifier.get_next_challenger(challengers=[self.config2, self.config1, self.config3],
                                                    chooser=None)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=None,
                                             run_history=self.rh,)

        self.assertEqual(inc, self.config2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)

        # run challenger now that the incumbent has been executed
        config, _ = intensifier.get_next_challenger(challengers=[self.config2, self.config1, self.config3],
                                                    chooser=None)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=inc,
                                             run_history=self.rh,)

        # challenger should have a better performance, so incumbent should have changed
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_DEFAULT)
        self.assertFalse(intensifier.continue_challenger)

        # run `always_race_against` now since the incumbent has changed
        config, _ = intensifier.get_next_challenger(challengers=[self.config2, self.config1, self.config3],
                                                    chooser=None)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=inc,
                                             run_history=self.rh,)

        self.assertEqual(inc, self.config1)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config3, only_max_observed_budget=True)), 1)
        self.assertEqual(intensifier.n_iters, 1)
        self.assertIsInstance(intensifier.configs_to_run, collections.Iterator)
        with self.assertRaises(StopIteration):
            next(intensifier.configs_to_run)
