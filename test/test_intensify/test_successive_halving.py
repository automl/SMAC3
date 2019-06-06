import unittest
from nose.plugins.attrib import attr

import logging
import numpy as np
import time

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.intensification.successive_halving import SuccessiveHalving
from smac.runhistory.runhistory import RunHistory, RunKey
from smac.optimizer.objective import average_cost, sum_cost
from smac.tae.execute_ta_run import StatusType
from smac.stats.stats import Stats
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


class TestSuccessiveHalving(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.rh = RunHistory(aggregate_func=average_cost)
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

    def test_compare_configs_chall(self):
        """
            challenger is better
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1])

        self.rh.add(config=self.config1, cost=1, time=2,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config2, cost=0, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        conf, _ = intensifier._compare_configs(incumbent=self.config1,
                                               challenger=self.config2,
                                               run_history=self.rh,
                                               aggregate_func=average_cost)

        # challenger has enough runs and is better
        self.assertEqual(conf, self.config2)

    def test_compare_configs_inc(self):
        """
            incumbent is better
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1])

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        conf, _ = intensifier._compare_configs(incumbent=self.config1,
                                               challenger=self.config2,
                                               run_history=self.rh,
                                               aggregate_func=average_cost)

        # challenger worse than inc
        self.assertEqual(conf, self.config1)

    def test_compare_configs_no_inc(self):
        """
            comparing configs with no incumbent run
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1])

        self.rh.add(config=self.config2, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=2,
                    seed=None,
                    additional_info=None)

        conf, _ = intensifier._compare_configs(incumbent=self.config1,
                                               challenger=self.config2,
                                               run_history=self.rh,
                                               aggregate_func=average_cost)

        # challenger config2 should be returned since there is no incumbent
        self.assertEqual(conf, self.config2)

    def test_compare_configs_unknow(self):
        """
            comparing configs with different sets of runs
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1])

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config1, cost=1, time=2,
                    status=StatusType.SUCCESS, instance_id=2,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=2,
                    seed=None,
                    additional_info=None)

        conf, _ = intensifier._compare_configs(incumbent=self.config1,
                                               challenger=self.config2,
                                               run_history=self.rh,
                                               aggregate_func=average_cost)

        # undecided - config is none
        self.assertIsNone(conf)

    def test_top_k(self):
        """
            test _top_k()
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1])
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=2,
                    seed=None,
                    additional_info=None)
        self.rh.add(config=self.config3, cost=10, time=10,
                    status=StatusType.SUCCESS, instance_id=3,
                    seed=None,
                    additional_info=None)
        conf = intensifier._top_k(configs=[self.config2, self.config1, self.config3],
                                  k=1, run_history=self.rh)

        self.assertEqual(conf, [self.config1])

    def test_adaptive_capping(self):
        """
            test _adapt_cutoff()
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=list(range(5)),
            deterministic=False)

        for i in range(5):
            self.rh.add(config=self.config1, cost=i + 1, time=i + 1,
                        status=StatusType.SUCCESS, instance_id=i,
                        seed=i,
                        additional_info=None)
        for i in range(3):
            self.rh.add(config=self.config2, cost=i + 1, time=i + 1,
                        status=StatusType.SUCCESS, instance_id=i,
                        seed=i,
                        additional_info=None)

        inst_seed_pairs = self.rh.get_runs_for_config(self.config1)
        # cost used by incumbent for going over all runs in inst_seed_pairs
        inc_sum_cost = sum_cost(config=self.config1, instance_seed_pairs=inst_seed_pairs,
                                run_history=self.rh)

        cutoff = intensifier._adapt_cutoff(challenger=self.config2,
                                           incumbent=self.config1,
                                           run_history=self.rh,
                                           inc_sum_cost=inc_sum_cost)
        # 15*1.2 - 6
        self.assertEqual(cutoff, 12)

        intensifier.cutoff = 5

        cutoff = intensifier._adapt_cutoff(challenger=self.config2,
                                           incumbent=self.config1,
                                           run_history=self.rh,
                                           inc_sum_cost=inc_sum_cost)
        # scenario cutoff
        self.assertEqual(cutoff, 5)

    @attr('slow')
    def test_intensify_1(self):
        """
           test intensify without adaptive capping
        """

        def target(x):
            return (x['a'] + 1) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True,
            instances=[1], min_budget=0.1, max_budget=1)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=0,
                    additional_info=None)

        inc, _ = intensifier.intensify(challengers=[self.config2],
                                       incumbent=self.config1,
                                       run_history=self.rh,
                                       aggregate_func=average_cost)

        self.assertEqual(inc, self.config2)

    @attr('slow')
    def test_intensify_2(self):
        """
           test intensify with adaptive capping & cutoff as budget
        """

        def target(x):
            time.sleep(1.5)
            return (x['a'] + 1) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime")
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True,
            instances=[1], min_budget=0.25, max_budget=1)

        self.rh.add(config=self.config1, cost=.001, time=0.001,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=0,
                    additional_info=None)

        # config2 should have a timeout (due to adaptive capping)
        # and config1 should still be the incumbent
        inc, _ = intensifier.intensify(challengers=[self.config2],
                                       incumbent=self.config1,
                                       run_history=self.rh,
                                       aggregate_func=average_cost)

        # self.assertTrue(False)
        self.assertEqual(inc, self.config1)

    @attr('slow')
    def test_intensify_3(self):
        """
            test intensify with multiple instance-seed pairs
        """

        def target(x: Configuration, seed: int, instance: str):
            if instance == 0:
                time.sleep(0.5)
            if x['b'] == 0:
                time.sleep(0.6)
            return (x['a'] + 1) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime")
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=False,
            instances=list(range(2)), n_seeds=2, cutoff=1, instance_order=None)

        # config1 should still be the incumbent - over all instance seed pairs
        inc, _ = intensifier.intensify(challengers=[self.config1, self.config2],
                                       incumbent=None,
                                       run_history=self.rh,
                                       aggregate_func=average_cost)

        self.assertEqual(inc, self.config1)

    @attr('slow')
    def test_intensify_4(self):
        """
            test intensify with solution quality as objective
        """

        def target(x: Configuration, seed: int, instance: str):
            return 1

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="quality")
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=list(range(2)),
            deterministic=True)

        for i in range(2):
            self.rh.add(config=self.config1, cost=i + 1, time=1,
                        status=StatusType.SUCCESS, instance_id=i,
                        seed=12345,
                        additional_info=None)

        # config2 should become the new incumbent since the objective is quality
        inc, _ = intensifier.intensify(challengers=[self.config2],
                                       incumbent=None,
                                       run_history=self.rh,
                                       aggregate_func=average_cost)

        self.assertEqual(inc, self.config2)
