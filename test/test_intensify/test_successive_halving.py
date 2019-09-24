import unittest
from nose.plugins.attrib import attr

import logging
import numpy as np
import time

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.intensification.successive_halving import SuccessiveHalving
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
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

    def test_init_1(self):
        """
            test parameter initializations for successive halving - instance as budget
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=False, run_obj_time=False,
            instances=[1, 2, 3], n_seeds=2, initial_budget=None, max_budget=None, eta=2)

        self.assertEqual(len(intensifier.instances), 6)  # since instance-seed pairs
        self.assertEqual(intensifier.initial_budget, 1)
        self.assertEqual(intensifier.max_budget, 6)
        self.assertEqual(intensifier.num_initial_challengers, 4)  # 2 iterations
        self.assertFalse(intensifier.cutoff_as_budget)

    def test_init_2(self):
        """
            test parameter initialiations for successive halving - cutoff as budget
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=30, instances=[1], initial_budget=1, max_budget=10, eta=2)

        self.assertEqual(len(intensifier.instances), 1)  # since instance-seed pairs
        self.assertEqual(intensifier.initial_budget, 1)
        self.assertEqual(intensifier.max_budget, 10)
        self.assertEqual(intensifier.num_initial_challengers, 8)  # 4 iterations
        self.assertTrue(intensifier.cutoff_as_budget)

    def test_init_3(self):
        """
            test parameter initialiations for successive halving - runtime cutoff as budget (no param provided)
        """

        with self.assertRaisesRegex(ValueError,
                                    "requires parameters initial_budget and max_budget/cutoff for intensification!"):
            SuccessiveHalving(
                tae_runner=None, stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1])

    def test_init_4(self):
        """
            test parameter initialiations for successive halving - eta < 1
        """

        with self.assertRaisesRegex(ValueError, 'eta must be greater than 1'):
            SuccessiveHalving(
                tae_runner=None, stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1], eta=0)

    def test_init_5(self):
        """
            test parameter initialiations for successive halving - eta < 1
        """

        with self.assertRaisesRegex(ValueError, 'eta must be greater than 1'):
            SuccessiveHalving(
                tae_runner=None, stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1], eta=0)

    def test_top_k_1(self):
        """
            test _top_k() for configs with same instance-seed-budget keys
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1], initial_budget=1)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=2,
                    seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=2,
                    seed=None,
                    additional_info=None)
        conf = intensifier._top_k(configs=[self.config2, self.config1],
                                  k=1, run_history=self.rh)

        self.assertEqual(conf, [self.config1])

    def test_top_k_2(self):
        """
            test _top_k() for configs with different instance-seed-budget keys
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1, 2], initial_budget=1)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=10, time=10,
                    status=StatusType.SUCCESS, instance_id=2,
                    seed=None,
                    additional_info=None)

        with self.assertRaisesRegex(AssertionError, 'Cannot compare configs'):
            intensifier._top_k(configs=[self.config2, self.config1, self.config3],
                               k=1, run_history=self.rh)

    @attr('slow')
    def test_intensify_1(self):
        """
           test intensify with quality objective & cutoff as budget
        """

        def target(x, instance, seed, budget):
            return (x['a'] + 1 + budget) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[1], initial_budget=1, max_budget=3, eta=2)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=0, budget=3,
                    additional_info=None)

        inc, _ = intensifier.intensify(challengers=[self.config2],
                                       incumbent=self.config1,
                                       run_history=self.rh,
                                       aggregate_func=average_cost)

        self.assertEqual(inc, self.config2)
        self.assertEqual(self.stats.ta_runs, 2)
        self.assertEqual(self.stats.inc_changed, 1)

    @attr('slow')
    def test_intensify_2(self):
        """
           test intensify with runtime objective and adaptive capping
        """

        def target(x):
            time.sleep(1.5)
            return (x['a'] + 1) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime")
        taf.runhistory = self.rh
        taf.runhistory.overwrite_existing_runs = True

        intensifier = SuccessiveHalving(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, cutoff=1,
            instances=[1, 2], initial_budget=1, max_budget=2, eta=2, instance_order=None)

        self.rh.add(config=self.config1, cost=.001, time=0.001,
                    status=StatusType.SUCCESS, instance_id=1, seed=0,
                    additional_info=None)
        self.rh.add(config=self.config1, cost=.001, time=0.001,
                    status=StatusType.SUCCESS, instance_id=2, seed=0,
                    additional_info=None)

        # config2 should be capped and config1 should still be the incumbent
        inc, _ = intensifier.intensify(challengers=[self.config2, self.config3],
                                       incumbent=self.config1,
                                       run_history=self.rh,
                                       aggregate_func=average_cost)

        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.ta_runs, 2)
        self.assertEqual(self.stats.inc_changed, 0)
        self.assertEqual(list(self.rh.data.values())[2][2], StatusType.CAPPED)
        self.assertEqual(list(self.rh.data.values())[3][2], StatusType.CAPPED)

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
            instances=[0, 1], n_seeds=2, cutoff=1, instance_order=None)

        # config1 should still be the incumbent - over all instance seed pairs
        inc, _ = intensifier.intensify(challengers=[self.config1, self.config2],
                                       incumbent=None,
                                       run_history=self.rh,
                                       aggregate_func=average_cost)

        self.assertEqual(inc, self.config1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config1)), 4)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2)), 1)

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
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=[0, 1],
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
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2)), 2)

    @attr('slow')
    def test_intensify_5(self):
        """
            test intensify with shuffling instance order every run
        """

        def target(x: Configuration, seed: int, instance: str):
            return 2*x['a'] + x['b'] + instance

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="quality")
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=[0, 1], instance_order='shuffle', eta=2,
            deterministic=True)

        # config1 should become the new incumbent
        inc, _ = intensifier.intensify(challengers=[self.config1, self.config2],
                                       incumbent=None,
                                       run_history=self.rh,
                                       aggregate_func=average_cost)

        self.assertEqual(inc, self.config1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2)), 1)

        # config1 stays the incumbent, but the previously rejected config2 is still executed
        inc, _ = intensifier.intensify(challengers=[self.config3, self.config2],
                                       incumbent=inc,
                                       run_history=self.rh,
                                       aggregate_func=average_cost)

        self.assertEqual(inc, self.config1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2)), 2)
