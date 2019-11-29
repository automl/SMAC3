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

        self.assertEqual(len(intensifier.inst_seed_pairs), 6)  # since instance-seed pairs
        self.assertEqual(len(intensifier.instances), 3)
        self.assertEqual(intensifier.initial_budget, 1)
        self.assertEqual(intensifier.max_budget, 6)
        self.assertListEqual(intensifier.n_configs_in_stage, [4.0, 2.0, 1.0])
        self.assertTrue(intensifier.instance_as_budget)
        self.assertTrue(intensifier.repeat_configs)

    def test_init_2(self):
        """
            test parameter initialiations for successive halving - real-valued budget
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[1], initial_budget=1, max_budget=10, eta=2)

        self.assertEqual(len(intensifier.inst_seed_pairs), 1)  # since instance-seed pairs
        self.assertEqual(intensifier.initial_budget, 1)
        self.assertEqual(intensifier.max_budget, 10)
        self.assertListEqual(intensifier.n_configs_in_stage, [8.0, 4.0, 2.0, 1.0])
        self.assertFalse(intensifier.instance_as_budget)
        self.assertFalse(intensifier.repeat_configs)

    def test_init_3(self):
        """
            test wrong parameter initializations for successive halving
        """

        # runtime as budget (no param provided)
        with self.assertRaisesRegex(ValueError,
                                    "requires parameters initial_budget and max_budget for intensification!"):
            SuccessiveHalving(
                tae_runner=None, stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1])

        # eta < 1
        with self.assertRaisesRegex(ValueError, "eta must be greater than 1"):
            SuccessiveHalving(
                tae_runner=None, stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1], eta=0)

        # max budget > instance-seed pairs
        with self.assertRaisesRegex(ValueError,
                                    "Max budget cannot be greater than the number of instance-seed pairs"):
            SuccessiveHalving(
                tae_runner=None, stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1, 2, 3], initial_budget=1, max_budget=5, n_seeds=1)

    def test_top_k_1(self):
        """
            test _top_k() for configs with same instance-seed-budget keys
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345),
            instances=[1], initial_budget=1)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=2, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=2, seed=None,
                    additional_info=None)
        conf = intensifier._top_k(configs=[self.config2, self.config1],
                                  k=1, run_history=self.rh)

        self.assertEqual(conf, [self.config1])

    def test_top_k_2(self):
        """
            test _top_k() for configs with different instance-seed-budget keys
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345),
            instances=[1, 2], initial_budget=1)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=10, time=10,
                    status=StatusType.SUCCESS, instance_id=2, seed=None,
                    additional_info=None)

        with self.assertRaisesRegex(AssertionError, 'Cannot compare configs'):
            intensifier._top_k(configs=[self.config2, self.config1, self.config3],
                               k=1, run_history=self.rh)

    def test_get_next_challenger_1(self):
        """
            test get_next_challenger for a presently running configuration
        """
        def target(x):
            return 1

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            tae_runner=taf, stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[1, 2], initial_budget=1, max_budget=2, eta=2)

        # next challenger from a list
        config, new = intensifier.get_next_challenger(challengers=[self.config1], chooser=None, run_history=self.rh)
        self.assertEqual(config, self.config1)
        self.assertTrue(new)

        # until evaluated, does not pick new challenger
        config, new = intensifier.get_next_challenger(challengers=[self.config2], chooser=None, run_history=self.rh)
        self.assertEqual(config, self.config1)
        self.assertEqual(intensifier.running_challenger, config)
        self.assertFalse(new)

        # evaluating configuration
        _ = intensifier.eval_challenger(challenger=config, incumbent=None, run_history=self.rh,
                                        aggregate_func=average_cost)
        config, new = intensifier.get_next_challenger(challengers=[self.config2], chooser=None, run_history=self.rh)
        self.assertEqual(config, self.config2)
        self.assertEqual(len(intensifier.curr_challengers), 1)
        self.assertTrue(new)

    def test_get_next_challenger_2(self):
        """
            test get_next_challenger for higher stages of SH iteration
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[1], initial_budget=1, max_budget=2, eta=2)

        intensifier._update_stage()
        intensifier.stage += 1
        intensifier.configs_to_run = [self.config1]

        # next challenger should come from configs to run
        config, new = intensifier.get_next_challenger(challengers=None, chooser=None, run_history=self.rh)
        self.assertEqual(config, self.config1)
        self.assertEqual(len(intensifier.configs_to_run), 0)
        self.assertFalse(new)

    def test_update_stage(self):
        """
            test update_stage - initializations for all tracking variables
        """
        intensifier = SuccessiveHalving(
            tae_runner=None, stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[1], initial_budget=1, max_budget=2, eta=2)

        # first stage update
        intensifier._update_stage()

        self.assertEqual(intensifier.stage, 0)
        self.assertEqual(intensifier.sh_iters, 0)
        self.assertEqual(intensifier.running_challenger, None)
        self.assertEqual(intensifier.curr_challengers, set())

        # higher stages
        self.rh.add(self.config1, 1, 1, StatusType.SUCCESS)
        self.rh.add(self.config2, 2, 2, StatusType.SUCCESS)
        intensifier.curr_challengers = {self.config1, self.config2}
        intensifier._update_stage(run_history=self.rh)

        self.assertEqual(intensifier.stage, 1)
        self.assertEqual(intensifier.sh_iters, 0)
        self.assertEqual(intensifier.configs_to_run, [self.config1])

        # next iteration
        intensifier.curr_challengers = {self.config1}
        intensifier._update_stage(run_history=self.rh)

        self.assertEqual(intensifier.stage, 0)
        self.assertEqual(intensifier.sh_iters, 1)
        self.assertEqual(intensifier.configs_to_run, None)

    @attr('slow')
    def test_eval_challenger_1(self):
        """
           test eval_challenger with quality objective & real-valued budget
        """

        def target(x: Configuration, instance: str, seed: int, budget: float):
            return 0.1 * budget

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[None], initial_budget=0.25, max_budget=0.5, eta=2)
        intensifier._update_stage()

        self.rh.add(config=self.config1, cost=1, time=1, status=StatusType.SUCCESS,
                    seed=0, budget=0.5)
        self.rh.add(config=self.config2, cost=1, time=1, status=StatusType.SUCCESS,
                    seed=0, budget=0.25)
        self.rh.add(config=self.config3, cost=2, time=1, status=StatusType.SUCCESS,
                    seed=0, budget=0.25)

        intensifier.curr_challengers = {self.config2, self.config3}
        intensifier._update_stage(run_history=self.rh)

        inc, inc_value = intensifier.eval_challenger(challenger=self.config2,
                                                     incumbent=self.config1,
                                                     run_history=self.rh,
                                                     aggregate_func=average_cost)

        self.assertEqual(inc, self.config2)
        self.assertEqual(inc_value, 0.05)
        self.assertEqual(self.stats.ta_runs, 1)
        self.assertEqual(list(self.rh.data.keys())[-1][0], self.rh.config_ids[self.config2])
        self.assertEqual(self.stats.inc_changed, 1)

    @attr('slow')
    def test_eval_challenger_2(self):
        """
           test eval_challenger with runtime objective and adaptive capping
        """

        def target(x):
            time.sleep(1.5)
            return (x['a'] + 1) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime")
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, cutoff=1,
            instances=[1, 2], initial_budget=1, max_budget=2, eta=2, instance_order=None)

        for i in range(2):
            self.rh.add(config=self.config1, cost=.001, time=0.001,
                        status=StatusType.SUCCESS, instance_id=i+1, seed=0,
                        additional_info=None)

        intensifier._update_stage()

        # config2 should be capped and config1 should still be the incumbent
        inc, _ = intensifier.eval_challenger(challenger=self.config2,
                                             incumbent=self.config1,
                                             run_history=self.rh,
                                             aggregate_func=average_cost)

        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.ta_runs, 1)
        self.assertEqual(self.stats.inc_changed, 0)
        self.assertEqual(list(self.rh.data.values())[2][2], StatusType.CAPPED)

    def test_eval_challenger_3(self):
        """
            test eval_challenger for updating to next stage and shuffling instance order every run
        """

        def target(x: Configuration, instance: str):
            return (x['a'] + int(instance)) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="quality")
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=[0, 1], instance_order='shuffle', eta=2,
            deterministic=True, cutoff=1)

        intensifier._update_stage()

        self.assertEqual(intensifier.inst_seed_pairs, [(0, 0), (1, 0)])

        config, _ = intensifier.get_next_challenger(challengers=[self.config1], chooser=None, run_history=self.rh)
        inc, _ = intensifier.eval_challenger(challenger=config,
                                             incumbent=None,
                                             run_history=self.rh,
                                             aggregate_func=average_cost)

        self.assertEqual(inc, None)   # since this is first run, doesn't return any incumbent
        self.assertEqual(len(self.rh.get_runs_for_config(self.config1)), 1)

        config, _ = intensifier.get_next_challenger(challengers=[self.config2], chooser=None, run_history=self.rh)
        inc, _ = intensifier.eval_challenger(challenger=config,
                                             incumbent=None,
                                             run_history=self.rh,
                                             aggregate_func=average_cost)

        self.assertEqual(inc, None)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2)), 1)
        self.assertEqual(intensifier.configs_to_run, [self.config1])
        self.assertEqual(intensifier.stage, 1)

        config, _ = intensifier.get_next_challenger(challengers=[self.config3], chooser=None, run_history=self.rh)
        inc, _ = intensifier.eval_challenger(challenger=config,
                                             incumbent=None,
                                             run_history=self.rh,
                                             aggregate_func=average_cost)

        self.assertEqual(inc, self.config1)  # run completed, incumbent generated

        self.assertEqual(len(self.rh.get_runs_for_config(self.config1)), 2)
        self.assertEqual(intensifier.sh_iters, 1)
        self.assertEqual(self.stats.inc_changed, 1)

        # For the 2nd SH iteration, we should still be able to run the old configurations again
        # since instance order is "shuffle"

        self.assertEqual(intensifier.inst_seed_pairs, [(1, 0), (0, 0)])

        config, _ = intensifier.get_next_challenger(challengers=[self.config2], chooser=None, run_history=self.rh)
        inc, _ = intensifier.eval_challenger(challenger=config,
                                             incumbent=None,
                                             run_history=self.rh,
                                             aggregate_func=average_cost)

        self.assertEqual(config, self.config2)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2)), 2)
