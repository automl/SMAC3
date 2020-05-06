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

    def test_eval_challenger_1(self):
        """
            test eval_challenger() - a complete intensification run with a `always_race_against` configuration
        """
        def target(x):
            if x['a'] == 100:
                time.sleep(1)
            return x['a']

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime")
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1, 2], run_obj_time=True, cutoff=2,
            deterministic=False, always_race_against=self.config3, run_limit=1)

        self.assertEqual(intensifier.n_iters, 0)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)

        # intensification iteration #1
        # run first config as incumbent if incumbent is None
        config, _ = intensifier.get_next_challenger(challengers=[self.config2],
                                                    chooser=None)
        self.assertEqual(config, self.config2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)
        # eval config 2 (=first run)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=None, run_history=self.rh, )
        self.assertEqual(inc, self.config2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(intensifier.n_iters, 1)  # 1 intensification run complete!

        # intensification iteration #2
        # regular intensification begins - run incumbent first
        config, _ = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                    chooser=None)
        self.assertEqual(config, inc)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=inc, run_history=self.rh, )
        self.assertEqual(self.stats.ta_runs, 2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(self.stats.inc_changed, 1)

        # run challenger now that the incumbent has been executed
        config, _ = intensifier.get_next_challenger(challengers=[self.config1],
                                                    chooser=None)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(config, self.config1)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=inc, run_history=self.rh, )

        # challenger has a better performance, but not run on all instances yet. so incumbent stays the same
        self.assertEqual(inc, self.config2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertTrue(intensifier.continue_challenger)

        # run challenger again on the other instance
        config, _ = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                    chooser=None)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(config, self.config1)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=inc, run_history=self.rh, )

        # challenger better than incumbent in both instances. so incumbent changed
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.inc_changed, 2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_BASIS)
        self.assertFalse(intensifier.continue_challenger)

        # run basis configuration (`always_race_against`)
        config, _ = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                    chooser=None)
        self.assertEqual(config, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_BASIS)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=inc, run_history=self.rh, )

        # the basis configuration (config3) not better than incumbent, so can move on
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.inc_changed, 2)
        self.assertEqual(self.stats.ta_runs, 5)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        self.assertEqual(list(self.rh.data.values())[4][2], StatusType.CAPPED)
        self.assertEqual(intensifier.n_iters, 1)  # iteration continues as `min_chall` condition is not met
        self.assertIsInstance(intensifier.configs_to_run, collections.abc.Iterator)
        # no configs should be left at the end
        with self.assertRaises(StopIteration):
            next(intensifier.configs_to_run)

        # intensification continues running incumbent again in same iteration...
        config, _ = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                    chooser=None)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=inc, run_history=self.rh, )

        self.assertEqual(inc, self.config1)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)

        self.assertEqual(len(self.rh.get_runs_for_config(self.config1, only_max_observed_budget=True)), 3)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2, only_max_observed_budget=True)), 2)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config3, only_max_observed_budget=True)), 0)  # capped

    def test_eval_challenger_2(self):
        """
            test eval_challenger() - a complete intensification run without a `always_race_against` configuration
        """
        def target(x):
            return 2 * x['a'] + x['b']

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="quality")
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1], run_obj_time=False,
            deterministic=True, always_race_against=None, run_limit=1)

        self.assertEqual(intensifier.n_iters, 0)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)

        # intensification iteration #1
        # run first config as incumbent if incumbent is None
        config, _ = intensifier.get_next_challenger(challengers=[self.config3],
                                                    chooser=None)
        self.assertEqual(config, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)
        # eval config 2 (=first run)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=None, run_history=self.rh, )
        self.assertEqual(inc, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(intensifier.n_iters, 1)  # 1 intensification run complete!

        # regular intensification begins - run incumbent
        config, _ = intensifier.get_next_challenger(challengers=None,  # since incumbent is run, no configs required
                                                    chooser=None)
        self.assertEqual(config, inc)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=inc, run_history=self.rh, )

        # no new TA runs as there are no more instances to run
        self.assertEqual(inc, self.config3)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(self.stats.ta_runs, 1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config3, only_max_observed_budget=True)), 1)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)

        # run challenger now that the incumbent has been executed
        config, _ = intensifier.get_next_challenger(challengers=[self.config2, self.config1],
                                                    chooser=None)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(config, self.config2)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=inc, run_history=self.rh, )

        # challenger has a better performance, so incumbent has changed
        self.assertEqual(inc, self.config2)
        self.assertEqual(self.stats.inc_changed, 2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)  # since there is no `always_race_against`
        self.assertFalse(intensifier.continue_challenger)
        self.assertEqual(intensifier.n_iters, 1)  # iteration continues as `min_chall` condition is not met

        # intensification continues running incumbent again in same iteration...
        # run incumbent
        config, _ = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                    chooser=None)
        self.assertEqual(config, self.config2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=inc, run_history=self.rh, )

        # run challenger
        config, _ = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                    chooser=None)
        self.assertEqual(config, self.config1)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=inc, run_history=self.rh, )

        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.inc_changed, 3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        self.assertEqual(intensifier.n_iters, 2)  # 2 intensification run complete!
        # no configs should be left at the end
        with self.assertRaises(StopIteration):
            next(intensifier.configs_to_run)

        self.assertEqual(len(self.rh.get_runs_for_config(self.config1, only_max_observed_budget=True)), 1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2, only_max_observed_budget=True)), 1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config3, only_max_observed_budget=True)), 1)

    def test_eval_challenger_3(self):
        """
            test eval_challenger for a resumed SMAC run (first run with incumbent)
        """
        def target(x):
            return x['a']

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="quality")
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), instances=[1], run_obj_time=False,
            deterministic=False, always_race_against=None, run_limit=1)

        self.assertEqual(intensifier.n_iters, 0)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)

        # adding run for incumbent configuration
        self.rh.add(config=self.config1, cost=1, time=1, status=StatusType.SUCCESS,
                    instance_id=1, seed=None, additional_info=None)

        # intensification - incumbent will be run, but not as RUN_FIRST_CONFIG stage
        config, _ = intensifier.get_next_challenger(challengers=[self.config2], chooser=None)
        inc, _ = intensifier.eval_challenger(challenger=config, incumbent=self.config1, run_history=self.rh, )

        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config1, only_max_observed_budget=True)), 2)
