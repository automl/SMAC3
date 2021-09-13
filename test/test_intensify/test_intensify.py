import collections
import unittest

import logging
import numpy as np
import time

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.intensification.abstract_racer import RunInfoIntent
from smac.runhistory.runhistory import RunHistory, RunInfo
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.intensification.intensification import Intensifier, IntensifierStage
from smac.facade.smac_ac_facade import SMAC4AC
from smac.tae import StatusType
from smac.utils.io.traj_logging import TrajLogger

from .test_eval_utils import eval_challenger

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


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

    def test_race_challenger_1(self):
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
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1], run_obj_time=False)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        intensifier.N = 1
        inc, instance, seed, cutoff = intensifier._get_next_racer(
            challenger=self.config2,
            incumbent=self.config1,
            run_history=self.rh
        )
        run_info = RunInfo(
            config=self.config2,
            instance=instance,
            instance_specific="0",
            cutoff=cutoff,
            seed=seed,
            capped=False,
            budget=0.0,
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
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
            stats=self.stats,
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
        inc, instance, seed, cutoff = intensifier._get_next_racer(
            challenger=self.config2,
            incumbent=self.config1,
            run_history=self.rh
        )
        run_info = RunInfo(
            config=self.config2,
            instance=instance,
            instance_specific="0",
            seed=seed,
            cutoff=cutoff,
            capped=True,
            budget=0.0,
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

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
            stats=self.stats,
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
        config, _ = intensifier.get_next_challenger(
            challengers=[self.config2, self.config3],
            chooser=None
        )
        inc, instance, seed, cutoff = intensifier._get_next_racer(
            challenger=config,
            incumbent=self.config1,
            run_history=self.rh
        )
        run_info = RunInfo(
            config=config,
            instance=instance,
            instance_specific="0",
            seed=seed,
            cutoff=cutoff,
            capped=True,
            budget=0.0,
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(inc, self.config1)

        # further run for incumbent
        self.rh.add(config=self.config1, cost=2, time=2,
                    status=StatusType.TIMEOUT, instance_id=2,
                    seed=12345,
                    additional_info=None)

        # give config2 a second chance - now it should run on both instances

        # run on instance 1
        config, _ = intensifier.get_next_challenger(challengers=[self.config2, self.config3], chooser=None)
        inc, instance, seed, cutoff = intensifier._get_next_racer(
            challenger=config,
            incumbent=self.config1,
            run_history=self.rh
        )
        run_info = RunInfo(
            config=config,
            instance=instance,
            instance_specific="0",
            seed=seed,
            cutoff=cutoff,
            capped=False,
            budget=0.0,
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        # run on instance 2
        config, _ = intensifier.get_next_challenger(challengers=[self.config3], chooser=None)
        self.assertEqual(config, self.config2)
        self.assertTrue(intensifier.continue_challenger)

        inc, instance, seed, cutoff = intensifier._get_next_racer(
            challenger=config,
            incumbent=self.config1,
            run_history=self.rh
        )
        run_info = RunInfo(
            config=config,
            instance=instance,
            instance_specific="0",
            seed=seed,
            cutoff=cutoff,
            capped=False,
            budget=0.0,
        )

        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        # the incumbent should still be config1 because
        # config2 should get on inst 1 a full timeout
        # such that c(config1) = 1.25 and c(config2) close to 1.3
        self.assertEqual(inc, self.config1)
        # the capped run should not be counted in runs_perf_config
        self.assertAlmostEqual(self.rh.num_runs_per_config[2], 2)
        self.assertFalse(intensifier.continue_challenger)

        self.assertEqual(intensifier.num_run, 3)
        self.assertEqual(intensifier.num_chall_run, 3)

    def test_race_challenger_large(self):
        """
           test _race_challenger using solution_quality
        """

        def target(x):
            return 1

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = Intensifier(
            stats=self.stats,
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
            if intensifier.continue_challenger:
                config = intensifier.current_challenger
            else:
                config, _ = intensifier.get_next_challenger(
                    challengers=[self.config2, self.config3],
                    chooser=None
                )
            inc, instance, seed, cutoff = intensifier._get_next_racer(
                challenger=config,
                incumbent=self.config1,
                run_history=self.rh
            )
            run_info = RunInfo(
                config=config,
                instance=instance,
                instance_specific="0",
                seed=seed,
                cutoff=cutoff,
                capped=False,
                budget=0.0,
            )

            result = eval_challenger(run_info, taf, self.stats, self.rh)
            inc, perf = intensifier.process_results(
                run_info=run_info,
                incumbent=self.config1,
                run_history=self.rh,
                time_bound=np.inf,
                result=result,
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
            stats=self.stats,
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
            if intensifier.continue_challenger:
                config = intensifier.current_challenger
            else:
                config, _ = intensifier.get_next_challenger(
                    challengers=[self.config2, self.config3],
                    chooser=None
                )
            inc, instance, seed, cutoff = intensifier._get_next_racer(
                challenger=config,
                incumbent=self.config1,
                run_history=self.rh
            )
            run_info = RunInfo(
                config=config,
                instance=instance,
                instance_specific="0",
                seed=seed,
                cutoff=cutoff,
                capped=False,
                budget=0.0,
            )
            result = eval_challenger(run_info, taf, self.stats, self.rh)
            inc, perf = intensifier.process_results(
                run_info=run_info,
                incumbent=self.config1,
                run_history=self.rh,
                time_bound=np.inf,
                result=result,
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
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1],
            deterministic=True)

        instance, seed, cutoff = intensifier._get_next_inc_run(
            available_insts=intensifier._get_inc_available_inst(
                incumbent=self.config1,
                run_history=self.rh
            )
        )
        run_info = RunInfo(
            config=self.config1,
            instance=instance,
            instance_specific="0",
            seed=seed,
            cutoff=cutoff,
            capped=False,
            budget=0.0,
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        intensifier.stage = IntensifierStage.PROCESS_FIRST_CONFIG_RUN
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(len(self.rh.data), 1, self.rh.data)

        # since we assume deterministic=1,
        # the second call should not add any more runs
        # given only one instance
        # So the returned seed/instance is None so that a new
        # run to be triggered is not launched
        available_insts = intensifier._get_inc_available_inst(
            incumbent=self.config1,
            run_history=self.rh
        )
        # Make sure that the list is empty, and hence no new call
        # of incumbent will be triggered
        self.assertFalse(available_insts)

        # The following two tests evaluate to zero because _next_iteration is triggered by _add_inc_run
        # as it is the first evaluation of this intensifier
        # After the above incumbent run, the stage is
        # IntensifierStage.RUN_CHALLENGER. Change it to test next iteration
        intensifier.stage = IntensifierStage.PROCESS_FIRST_CONFIG_RUN
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(intensifier.num_run, 0)
        self.assertEqual(intensifier.num_chall_run, 0)

    def test_add_inc_run_nondet(self):
        """
            test _add_inc_run()
        """

        def target(x):
            return (x['a'] + 1) / 1000.
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="solution_quality")

        intensifier = Intensifier(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1, 2],
            deterministic=False)

        instance, seed, cutoff = intensifier._get_next_inc_run(
            available_insts=intensifier._get_inc_available_inst(
                incumbent=self.config1,
                run_history=self.rh
            )
        )
        run_info = RunInfo(
            config=self.config1,
            instance=instance,
            instance_specific="0",
            seed=seed,
            cutoff=cutoff,
            capped=False,
            budget=0.0,
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(len(self.rh.data), 1, self.rh.data)

        instance, seed, cutoff = intensifier._get_next_inc_run(
            available_insts=intensifier._get_inc_available_inst(
                incumbent=self.config1,
                run_history=self.rh
            )
        )
        run_info = RunInfo(
            config=self.config1,
            instance=instance,
            instance_specific="0",
            seed=seed,
            cutoff=cutoff,
            capped=False,
            budget=0.0,
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(len(self.rh.data), 2, self.rh.data)
        runs = self.rh.get_runs_for_config(config=self.config1, only_max_observed_budget=True)
        # exactly one run on each instance
        self.assertIn(1, [runs[0].instance, runs[1].instance])
        self.assertIn(2, [runs[0].instance, runs[1].instance])

        instance, seed, cutoff = intensifier._get_next_inc_run(
            available_insts=intensifier._get_inc_available_inst(
                incumbent=self.config1,
                run_history=self.rh
            )
        )
        run_info = RunInfo(
            config=self.config1,
            instance=instance,
            instance_specific="0",
            seed=seed,
            cutoff=cutoff,
            capped=False,
            budget=0.0,
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(len(self.rh.data), 3, self.rh.data)

        # The number of runs performed should be 3
        # No Next iteration call as an incumbent is provided
        self.assertEqual(intensifier.num_run, 2)
        self.assertEqual(intensifier.num_chall_run, 0)

    def testget_next_challenger(self):
        """
            test get_next_challenger()
        """
        intensifier = Intensifier(
            stats=self.stats,
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
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), instances=[1],
            deterministic=True)

        gen = intensifier._generate_challengers(challengers=[self.config1, self.config2], chooser=None)

        self.assertEqual(next(gen), self.config1)
        self.assertEqual(next(gen), self.config2)
        self.assertRaises(StopIteration, next, gen)

        # test get generator from a chooser - would return only 1 configuration
        intensifier = Intensifier(
            stats=self.stats, traj_logger=None,
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
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1, 2], run_obj_time=True, cutoff=2,
            deterministic=False, always_race_against=self.config3, run_limit=1)

        self.assertEqual(intensifier.n_iters, 0)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)

        # intensification iteration #1
        # run first config as incumbent if incumbent is None
        intent, run_info = intensifier.get_next_run(
            incumbent=None,
            run_history=self.rh,
            challengers=[self.config2],
            chooser=None
        )
        self.assertEqual(run_info.config, self.config2)
        self.assertEqual(intensifier.stage, IntensifierStage.PROCESS_FIRST_CONFIG_RUN)
        # eval config 2 (=first run)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(inc, self.config2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(intensifier.n_iters, 1)  # 1 intensification run complete!

        # intensification iteration #2
        # regular intensification begins - run incumbent first
        intent, run_info = intensifier.get_next_run(
            challengers=None,  # don't need a new list here as old one is cont'd
            incumbent=inc,
            run_history=self.rh,
            chooser=None
        )
        self.assertEqual(run_info.config, inc)
        self.assertEqual(intensifier.stage, IntensifierStage.PROCESS_INCUMBENT_RUN)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(self.stats.inc_changed, 1)

        # run challenger now that the incumbent has been executed
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1],
            incumbent=inc,
            run_history=self.rh,
            chooser=None
        )
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(run_info.config, self.config1)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        # challenger has a better performance, but not run on all instances yet. so incumbent stays the same
        self.assertEqual(inc, self.config2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertTrue(intensifier.continue_challenger)

        # run challenger again on the other instance
        intent, run_info = intensifier.get_next_run(
            challengers=None,  # don't need a new list here as old one is cont'd
            incumbent=inc,
            run_history=self.rh,
            chooser=None
        )
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(run_info.config, self.config1)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        # challenger better than incumbent in both instances. so incumbent changed
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.inc_changed, 2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_BASIS)
        self.assertFalse(intensifier.continue_challenger)

        # run basis configuration (`always_race_against`)
        intent, run_info = intensifier.get_next_run(
            challengers=None,  # don't need a new list here as old one is cont'd
            incumbent=inc,
            run_history=self.rh,
            chooser=None
        )
        self.assertEqual(run_info.config, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_BASIS)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        # the basis configuration (config3) not better than incumbent, so can move on
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.inc_changed, 2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT,
                         msg=self.rh.data.items())
        self.assertEqual(list(self.rh.data.values())[4][2], StatusType.CAPPED)
        self.assertEqual(intensifier.n_iters, 1)  # iteration continues as `min_chall` condition is not met
        self.assertIsInstance(intensifier.configs_to_run, collections.abc.Iterator)
        # no configs should be left at the end
        with self.assertRaises(StopIteration):
            next(intensifier.configs_to_run)

        # intensification continues running incumbent again in same iteration...
        intent, run_info = intensifier.get_next_run(
            challengers=None,  # don't need a new list here as old one is cont'd
            incumbent=inc,
            run_history=self.rh,
            chooser=None
        )
        self.assertEqual(intensifier.stage, IntensifierStage.PROCESS_INCUMBENT_RUN)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
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
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1], run_obj_time=False,
            deterministic=True, always_race_against=None, run_limit=1)

        self.assertEqual(intensifier.n_iters, 0)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)

        # intensification iteration #1
        # run first config as incumbent if incumbent is None
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config3],
            incumbent=None,
            run_history=self.rh,
            chooser=None
        )
        self.assertEqual(run_info.config, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.PROCESS_FIRST_CONFIG_RUN)
        # eval config 2 (=first run)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(intensifier.n_iters, 1)  # 1 intensification run complete!

        # regular intensification begins - run incumbent
        # Normally a challenger will be given, which in this case is the incumbent
        # But no more instances are available. So to prevent cicles
        # where No iteration happens, provide the challengers
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2, self.config1],  # since incumbent is run, no configs required
            incumbent=inc,
            run_history=self.rh,
            chooser=None
        )

        # no new TA runs as there are no more instances to run
        self.assertEqual(inc, self.config3)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config3, only_max_observed_budget=True)), 1)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)

        # run challenger now that the incumbent has been executed
        # So this call happen above, to save one iteration
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(run_info.config, self.config2)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        # challenger has a better performance, so incumbent has changed
        self.assertEqual(inc, self.config2)
        self.assertEqual(self.stats.inc_changed, 2)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)  # since there is no `always_race_against`
        self.assertFalse(intensifier.continue_challenger)
        self.assertEqual(intensifier.n_iters, 1)  # iteration continues as `min_chall` condition is not met

        # intensification continues running incumbent again in same iteration...
        # run incumbent
        # Same here, No further instance-seed pairs for incumbent available
        # so above call gets the new config to run
        self.assertEqual(run_info.config, self.config2)

        # There is a transition from:
        # IntensifierStage.RUN_FIRST_CONFIG-> IntensifierStage.RUN_INCUMBENT
        # Because after the first run, incumbent is run.
        # Nevertheless, there is now a transition:
        # IntensifierStage.RUN_INCUMBENT->IntensifierStage.RUN_CHALLENGER
        # because in add_inc_run, there are more available instance pairs
        # FROM: IntensifierStage.RUN_INCUMBENT TO: IntensifierStage.RUN_INCUMBENT WHY: no more to run
        # if all <instance, seed> have been run, compare challenger performance
        # self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)

        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        # run challenger
        intent, run_info = intensifier.get_next_run(
            challengers=None,  # don't need a new list here as old one is cont'd
            incumbent=inc,
            run_history=self.rh,
            chooser=None
        )
        self.assertEqual(run_info.config, self.config1)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
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
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), instances=[1], run_obj_time=False,
            deterministic=False, always_race_against=None, run_limit=1)

        self.assertEqual(intensifier.n_iters, 0)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)

        # adding run for incumbent configuration
        self.rh.add(config=self.config1, cost=1, time=1, status=StatusType.SUCCESS,
                    instance_id=1, seed=None, additional_info=None)

        # intensification - incumbent will be run, but not as RUN_FIRST_CONFIG stage
        intent_, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            incumbent=self.config1,
            run_history=self.rh,
            chooser=None
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
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
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1], run_obj_time=False,
            deterministic=True, always_race_against=None, run_limit=1,
            min_chall=1,
        )

        self.assertEqual(intensifier.n_iters, 0)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_FIRST_CONFIG)

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config3],
            incumbent=None,
            run_history=self.rh,
            chooser=None
        )
        self.assertEqual(run_info.config, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.PROCESS_FIRST_CONFIG_RUN)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config3)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_INCUMBENT)
        self.assertEqual(intensifier.n_iters, 1)  # 1 intensification run complete!

        # regular intensification begins - run incumbent

        # No further instance-seed pairs for incumbent available
        # Here None challenger is suggested. Code jumps to next iteration
        # This causes a transition from RUN_INCUMBENT->RUN_CHALLENGER
        # But then, the next configuration to run is the incumbent
        # We don't rerun the incumbent (see message):
        # Challenger was the same as the current incumbent; Skipping challenger
        # Then, we try to get more challengers, but below all challengers
        # Provided are config3, the incumbent which means nothing more to run
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config3],  # since incumbent is run, no configs required
            incumbent=inc,
            run_history=self.rh,
            chooser=None
        )

        self.assertEqual(run_info.config, None)
        self.assertEqual(intensifier.stage, IntensifierStage.RUN_CHALLENGER)

        intensifier._next_iteration()

        # Add a configuration, then try to execute it afterwards
        self.assertEqual(intensifier.n_iters, 2)

        self.rh.add(config=self.config1, cost=1, time=1, status=StatusType.SUCCESS,
                    instance_id=1, seed=0, additional_info=None)
        intensifier.stage = IntensifierStage.RUN_CHALLENGER

        # In the upcoming get next run, the stage is RUN_CHALLENGER
        # so the intensifier tries to run config1. Nevertheless,
        # there are no further instances for this configuration available.
        # In this scenario, the intensifier produces a SKIP intent as an indication
        # that a new iteration must be initiated, and for code simplicity,
        # relies on a new call to get_next_run to yield more configurations
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1],
            incumbent=inc,
            run_history=self.rh,
            chooser=None
        )
        self.assertEqual(intent, RunInfoIntent.SKIP)

        # This doesn't return a config because the array of configs is exhausted
        intensifier.stage = IntensifierStage.RUN_CHALLENGER
        config, _ = intensifier.get_next_challenger(challengers=None,
                                                    chooser=None)
        self.assertIsNone(config)
        # This finally gives a runable configuration
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            incumbent=inc,
            run_history=self.rh,
            chooser=None
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, perf = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        # 4 Iterations due to the proactive runs
        # of get next challenger
        self.assertEqual(intensifier.n_iters, 3)
        self.assertEqual(intensifier.num_chall_run, 1)
