import collections
import copy
import unittest

import logging
import numpy as np
import time

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.runhistory.runhistory import RunHistory, RunInfo
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
           Makes sure that a racing configuration with better performance,
           is selected as incumbent
           No adaptive capping
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

        # For Race challenger to be called, the intensifier has
        # to be in RUN_CHALLENGER STAGE. Also, the substatus
        # cariable intensifier.to_run_empty is set to true
        # which means that all challenger configurations are ready
        # to be compared
        intensifier.N = 1
        intensifier.stage = IntensifierStage.RUN_CHALLENGER
        intensifier.to_run_empty = True
        inc, instance, seed, cutoff = intensifier._race_challenger(
            challenger=self.config2,
            incumbent=self.config1,
            run_history=self.rh
        )
        status, cost, dur, res = intensifier.eval_challenger(
            RunInfo(
                config=self.config2,
                new=True,
                instance=instance,
                seed=seed,
                cutoff=cutoff,
                budget=0.0,
            )
        )
        inc, perf = intensifier.process_results(
            challenger=self.config2,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

        self.assertEqual(inc, self.config2)
        self.assertEqual(intensifier.num_run, 1)
        self.assertEqual(intensifier.num_chall_run, 1)

    def test_race_challenger_2(self):
        """
           Makes sure that a racing configuration with better performance,
           that is capped, doesn't substitute the incumbent.
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

        # For Race challenger to be called, the intensifier has
        # to be in RUN_CHALLENGER STAGE. Also, the substatus
        # cariable intensifier.to_run_empty is set to true
        # which means that all challenger configurations are ready
        # to be compared
        intensifier.N = 1
        intensifier.stage = IntensifierStage.RUN_CHALLENGER
        intensifier.to_run_empty = True
        inc, instance, seed, cutoff = intensifier._race_challenger(
            challenger=self.config2,
            incumbent=self.config1,
            run_history=self.rh
        )
        status, cost, dur, res = intensifier.eval_challenger(
            RunInfo(
                config=self.config2,
                new=True,
                instance=instance,
                seed=seed,
                cutoff=cutoff,
                budget=0.0,
            )
        )
        inc, perf = intensifier.process_results(
            challenger=self.config2,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

        # config2 should have a timeout (due to adaptive capping)
        # and config1 should still be the incumbent
        self.assertEqual(inc, self.config1)
        self.assertEqual(intensifier.num_run, 1)
        self.assertEqual(intensifier.num_chall_run, 1)

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
        run_info = intensifier.get_next_challenger(
            challengers=[self.config2, self.config3],
            incumbent=self.config1,
            run_history=self.rh,
            chooser=None
        )
        inc, instance, seed, cutoff = intensifier._race_challenger(
            challenger=run_info.config,
            incumbent=self.config1,
            run_history=self.rh
        )
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=0,
        )

        self.assertEqual(inc, self.config1)

        # further run for incumbent
        self.rh.add(config=self.config1, cost=2, time=2,
                    status=StatusType.TIMEOUT, instance_id=2,
                    seed=12345,
                    additional_info=None)

        # give config2 a second chance - now it should run on both instances

        # run on instance 1

        run_info = intensifier.get_next_challenger(
            challengers=[self.config2, self.config3],
            incumbent=self.config1,
            run_history=self.rh,
            chooser=None
        )
        inc, instance, seed, cutoff = intensifier._race_challenger(
            challenger=run_info.config,
            incumbent=self.config1,
            run_history=self.rh
        )
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=0,
        )

        # run on instance 2
        run_info = intensifier.get_next_challenger(
            challengers=[self.config3],
            incumbent=self.config1,
            run_history=self.rh,
            chooser=None
        )

        # Because the run is capped, the stage do a forced transition to
        # IntensifierStage.RUN_INCUMBENT. So the challenger to run has to
        # be the incumbent
        self.assertEqual(run_info.config, self.config1)
        self.assertFalse(intensifier.continue_challenger)

        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=0,
        )

        # Add a test to make sure continue challenger is set to True
        # If comparing against config 2, not enough runs have been
        # performed. So continue challenger is true so that a new
        # run is encouraged
        inc, perf = intensifier.process_results(
            challenger=self.config2,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=0,
        )
        self.assertTrue(intensifier.continue_challenger)

        # the incumbent should still be config1 because
        # config2 should get on inst 1 a full timeout
        # such that c(config1) = 1.25 and c(config2) close to 1.3
        self.assertEqual(inc, self.config1)

        # the capped run should not be counted in runs_perf_config
        # This is something handled by tae runner
        num_runs_per_config_before = copy.deepcopy(self.rh.num_runs_per_config)
        status, cost, dur, res = intensifier.eval_challenger(
            RunInfo(
                config=self.config2,
                new=True,
                instance=1,
                seed=12345,
                cutoff=0.5,
                budget=0.0,
            )
        )
        self.assertEqual(status, StatusType.CAPPED)
        self.assertDictEqual(self.rh.num_runs_per_config, num_runs_per_config_before)

        self.assertEqual(intensifier.num_run, 3)

        # On capped runs, the incumbent is run. We do not expect
        # 3 challenger runs to have executed, just 1
        self.assertEqual(intensifier.num_chall_run, 1)

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
            run_info = intensifier.get_next_challenger(
                challengers=[self.config2, self.config3],
                incumbent=self.config1,
                run_history=self.rh,
                chooser=None
            )
            status, cost, dur, res = intensifier.eval_challenger(run_info)
            inc, perf = intensifier.process_results(
                challenger=run_info.config,
                incumbent=self.config1,
                run_history=self.rh,
                time_bound=np.inf,
                status=status,
                runtime=dur,
                elapsed_time=0,
            )

            # stop when challenger evaluation is over
            if not intensifier.stage == IntensifierStage.RUN_CHALLENGER:
                break

        self.assertEqual(inc, self.config2)
        self.assertEqual(self.rh.get_cost(self.config2), 1)

        # get data for config2 to check that the correct run was performed
        runs = self.rh.get_runs_for_config(self.config2, only_max_observed_budget=True)
        self.assertEqual(len(runs), 10)

        self.assertEqual(intensifier.num_run, 10)
        self.assertEqual(intensifier.num_chall_run, 10)

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
            run_info = intensifier.get_next_challenger(
                challengers=[self.config2, self.config3],
                incumbent=self.config1,
                run_history=self.rh,
                chooser=None
            )
            status, cost, dur, res = intensifier.eval_challenger(run_info)
            inc, perf = intensifier.process_results(
                challenger=run_info.config,
                incumbent=self.config1,
                run_history=self.rh,
                time_bound=np.inf,
                status=status,
                runtime=dur,
                elapsed_time=0,
            )

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

        self.assertEqual(intensifier.num_run, 10)
        self.assertEqual(intensifier.num_chall_run, 10)

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

        inc, instance, seed, cutoff = intensifier._add_inc_run(
            incumbent=self.config1,
            run_history=self.rh
        )
        status, cost, dur, res = intensifier.eval_challenger(
            RunInfo(
                config=self.config1,
                new=False,
                instance=instance,
                seed=seed,
                cutoff=cutoff,
                budget=0.0,
            )
        )
        inc, perf = intensifier.process_results(
            challenger=self.config1,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(len(self.rh.data), 1, self.rh.data)

        # since we assume deterministic=1,
        # the second call should not add any more runs
        # given only one instance
        # So the returned seed/instance is None so that a new
        # run to be triggered is not launched
        inc, instance, seed, cutoff = intensifier._add_inc_run(
            incumbent=self.config1,
            run_history=self.rh
        )
        self.assertEqual(None, instance)
        self.assertEqual(None, seed)

        # The following two tests evaluate to zero because _next_iteration is triggered by _add_inc_run
        # as it is the first evaluation of this intensifier
        # After the above incumbent run, the stage is
        # IntensifierStage.RUN_CHALLENGER. Change it to test next iteration
        intensifier.stage = IntensifierStage.RUN_FIRST_CONFIG
        inc, perf = intensifier.process_results(
            challenger=self.config1,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(intensifier.num_run, 0)
        self.assertEqual(intensifier.num_chall_run, 0)

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

        inc, instance, seed, cutoff = intensifier._add_inc_run(
            incumbent=self.config1,
            run_history=self.rh
        )
        status, cost, dur, res = intensifier.eval_challenger(
            RunInfo(
                config=self.config1,
                new=False,
                instance=instance,
                seed=seed,
                cutoff=cutoff,
                budget=0.0,
            )
        )
        inc, perf = intensifier.process_results(
            challenger=self.config1,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(len(self.rh.data), 1, self.rh.data)

        inc, instance, seed, cutoff = intensifier._add_inc_run(
            incumbent=self.config1,
            run_history=self.rh
        )
        status, cost, dur, res = intensifier.eval_challenger(
            RunInfo(
                config=self.config1,
                new=False,
                instance=instance,
                seed=seed,
                cutoff=cutoff,
                budget=0.0,
            )
        )
        inc, perf = intensifier.process_results(
            challenger=self.config1,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(len(self.rh.data), 2, self.rh.data)
        runs = self.rh.get_runs_for_config(config=self.config1, only_max_observed_budget=True)
        # exactly one run on each instance
        self.assertIn(1, [runs[0].instance, runs[1].instance])
        self.assertIn(2, [runs[0].instance, runs[1].instance])

        inc, instance, seed, cutoff = intensifier._add_inc_run(
            incumbent=self.config1,
            run_history=self.rh
        )
        status, cost, dur, res = intensifier.eval_challenger(
            RunInfo(
                config=self.config1,
                new=False,
                instance=instance,
                seed=seed,
                cutoff=cutoff,
                budget=0.0,
            )
        )
        inc, perf = intensifier.process_results(
            challenger=self.config1,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(len(self.rh.data), 3, self.rh.data)

        # The number of runs performed should be 3
        # No Next iteration call as an incumbent is provided
        self.assertEqual(intensifier.num_run, 3)
        self.assertEqual(intensifier.num_chall_run, 0)

    def test_get_next_challenger(self):
        """
            test get_next_challenger()
        """
        intensifier = Intensifier(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1],
            deterministic=True,
            run_obj_time=False,
        )

        intensifier.stage = IntensifierStage.RUN_CHALLENGER

        # get a new challenger to evaluate
        run_info = intensifier.get_next_challenger(
            challengers=[self.config1, self.config2],
            run_history=self.rh,
            incumbent=self.config2,
            chooser=None
        )

        self.assertEqual(run_info.config, self.config1, intensifier.current_challenger)
        self.assertEqual(intensifier._chall_indx, 1)
        self.assertEqual(intensifier.N, 1)
        self.assertTrue(run_info.new)

        # when already evaluating a challenger, return the same challenger
        intensifier.to_run = [(1, 1, 0)]
        run_info = intensifier.get_next_challenger(
            challengers=[self.config2],
            run_history=self.rh,
            incumbent=self.config1,
            chooser=None,
        )
        self.assertEqual(self.config1, intensifier.current_challenger)

        # During evaluation, if the challenger is the same as the
        # Incumbent, the challenger is skipped, if the stage is
        # Running challenger. In this case None is returned as next config
        self.assertEqual(run_info.config, None)
        self.assertEqual(intensifier._chall_indx, 1)
        self.assertFalse(run_info.new)

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
        run_info = intensifier.get_next_challenger(
            incumbent=None,
            run_history=self.rh,
            challengers=[self.config2],
            chooser=None
        )
        self.assertEqual(run_info.config, self.config2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)
        # eval config 2 (=first run)
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

        self.assertEqual(inc, self.config2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(intensifier.n_iters, 1)  # 1 intensification run complete!

        # intensification iteration #2
        # regular intensification begins - run incumbent first
        run_info = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(run_info.config, inc)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(self.stats.ta_runs, 2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(self.stats.inc_changed, 1)

        # run challenger now that the incumbent has been executed
        run_info = intensifier.get_next_challenger(challengers=[self.config1],
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(run_info.config, self.config1)
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

        # challenger has a better performance, but not run on all instances yet. so incumbent stays the same
        self.assertEqual(inc, self.config2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertTrue(intensifier.continue_challenger)

        # run challenger again on the other instance
        run_info = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(run_info.config, self.config1)
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

        # challenger better than incumbent in both instances. so incumbent changed
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.inc_changed, 2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_BASIS)
        self.assertFalse(intensifier.continue_challenger)

        # run basis configuration (`always_race_against`)
        run_info = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(run_info.config, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_BASIS)
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

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
        run_info = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

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
        run_info = intensifier.get_next_challenger(challengers=[self.config3],
                                                   incumbent=None,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(run_info.config, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)
        # eval config 2 (=first run)
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(inc, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(intensifier.n_iters, 1)  # 1 intensification run complete!

        # regular intensification begins - run incumbent
        run_info = intensifier.get_next_challenger(challengers=None,  # since incumbent is run, no configs required
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(run_info.config, inc)

        # There is a transition from:
        # IntensifierStage.RUN_FIRST_CONFIG-> IntensifierStage.RUN_INCUMBENT
        # Because after the first run, incumbent is run.
        # Nevertheless, there is now a transition:
        # IntensifierStage.RUN_INCUMBENT->IntensifierStage.RUN_CHALLENGER
        # because in add_inc_run, there are more available instance pairs
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        if run_info.config and (run_info.instance is not None or run_info.seed is not None):
            status, cost, dur, res = intensifier.eval_challenger(run_info)
        else:
            status, cost, dur, res = None, None, None, None
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

        # no new TA runs as there are no more instances to run
        self.assertEqual(inc, self.config3)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(self.stats.ta_runs, 1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config3, only_max_observed_budget=True)), 1)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)

        # run challenger now that the incumbent has been executed
        run_info = intensifier.get_next_challenger(challengers=[self.config2, self.config1],
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(run_info.config, self.config2)
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

        # challenger has a better performance, so incumbent has changed
        self.assertEqual(inc, self.config2)
        self.assertEqual(self.stats.inc_changed, 2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)  # since there is no `always_race_against`
        self.assertFalse(intensifier.continue_challenger)
        self.assertEqual(intensifier.n_iters, 1)  # iteration continues as `min_chall` condition is not met

        # intensification continues running incumbent again in same iteration...
        # run incumbent
        run_info = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(run_info.config, self.config2)

        # There is a transition from:
        # IntensifierStage.RUN_FIRST_CONFIG-> IntensifierStage.RUN_INCUMBENT
        # Because after the first run, incumbent is run.
        # Nevertheless, there is now a transition:
        # IntensifierStage.RUN_INCUMBENT->IntensifierStage.RUN_CHALLENGER
        # because in add_inc_run, there are more available instance pairs
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)

        if run_info.config and (run_info.instance is not None or run_info.seed is not None):
            status, cost, dur, res = intensifier.eval_challenger(run_info)
        else:
            status, cost, dur, res = None, None, None, None
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

        # run challenger
        run_info = intensifier.get_next_challenger(challengers=None,  # don't need a new list here as old one is cont'd
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(run_info.config, self.config1)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

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
        run_info = intensifier.get_next_challenger(challengers=[self.config2],
                                                   incumbent=self.config1,
                                                   run_history=self.rh,
                                                   chooser=None)
        status, cost, dur, res = intensifier.eval_challenger(run_info)
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )

        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config1, only_max_observed_budget=True)), 2)

    def test_no_new_intensification_wo_challenger_run(self):
        """
        This test ensures that no new iteration is started if no challenger run was conducted
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
            deterministic=True, always_race_against=None, run_limit=1,
            min_chall=1,
        )

        self.assertEqual(intensifier.n_iters, 0)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)

        run_info = intensifier.get_next_challenger(challengers=[self.config3],
                                                   incumbent=None,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(run_info.config, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)
        if run_info.config and (run_info.instance is not None or run_info.seed is not None):
            status, cost, dur, res = intensifier.eval_challenger(run_info)
        else:
            status, cost, dur, res = None, None, None, None
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(inc, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        self.assertEqual(intensifier.n_iters, 1)  # 1 intensification run complete!

        # regular intensification begins - run incumbent
        run_info = intensifier.get_next_challenger(challengers=None,  # since incumbent is run, no configs required
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertEqual(run_info.config, inc)
        # There is a transition from:
        # IntensifierStage.RUN_FIRST_CONFIG-> IntensifierStage.RUN_INCUMBENT
        # Because after the first run, incumbent is run.
        # Nevertheless, there is now a transition:
        # IntensifierStage.RUN_INCUMBENT->IntensifierStage.RUN_CHALLENGER
        # because in add_inc_run, there are more available instance pairs
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        if run_info.config and (run_info.instance is not None or run_info.seed is not None):
            status, cost, dur, res = intensifier.eval_challenger(run_info)
        else:
            status, cost, dur, res = None, None, None, None
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(intensifier.n_iters, 1)

        # Check that we don't walk into the next iteration if the challenger is passed again
        run_info = intensifier.get_next_challenger(challengers=[self.config3],
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        if run_info.config and (run_info.instance is not None or run_info.seed is not None):
            status, cost, dur, res = intensifier.eval_challenger(run_info)
        else:
            status, cost, dur, res = None, None, None, None
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(intensifier.n_iters, 1)

        intensifier._next_iteration()

        # Add a configuration, then try to execute it afterwards
        self.assertEqual(intensifier.n_iters, 2)
        self.rh.add(config=self.config1, cost=1, time=1, status=StatusType.SUCCESS,
                    instance_id=1, seed=0, additional_info=None)
        intensifier.stage = IntensifierStage.RUN_CHALLENGER
        run_info = intensifier.get_next_challenger(challengers=[self.config1],
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        if run_info.config and (run_info.instance is not None or run_info.seed is not None):
            status, cost, dur, res = intensifier.eval_challenger(run_info)
        else:
            status, cost, dur, res = None, None, None, None
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(intensifier.n_iters, 2)
        self.assertEqual(intensifier.num_chall_run, 0)

        # This returns the config evaluating the incumbent again
        run_info = intensifier.get_next_challenger(challengers=None,
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        if run_info.config and (run_info.instance is not None or run_info.seed is not None):
            status, cost, dur, res = intensifier.eval_challenger(run_info)
        else:
            status, cost, dur, res = None, None, None, None
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        # This doesn't return a config because the array of configs is exhausted
        run_info = intensifier.get_next_challenger(challengers=None,
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        self.assertIsNone(run_info.config)
        # This finally gives a runable configuration
        run_info = intensifier.get_next_challenger(challengers=[self.config2],
                                                   incumbent=inc,
                                                   run_history=self.rh,
                                                   chooser=None)
        if run_info.config and (run_info.instance is not None or run_info.seed is not None):
            status, cost, dur, res = intensifier.eval_challenger(run_info)  # noqa: F841
        else:
            status, cost, dur, res = None, None, None, None  # noqa: F841
        inc, perf = intensifier.process_results(
            challenger=run_info.config,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            status=status,
            runtime=dur,
            elapsed_time=dur,
        )
        self.assertEqual(intensifier.n_iters, 3)
        self.assertEqual(intensifier.num_chall_run, 1)
