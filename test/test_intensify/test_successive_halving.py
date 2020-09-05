import unittest
from unittest import mock

import logging
import numpy as np
import time

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.intensification.abstract_racer import RunInfoIntent
from smac.intensification.successive_halving import SuccessiveHalving
from smac.runhistory.runhistory import RunHistory
from smac.tae import StatusType
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger

from .test_eval_utils import eval_challenger


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

        self.rh = RunHistory()
        self.cs = get_config_space()
        self.config1 = Configuration(self.cs,
                                     values={'a': 0, 'b': 100})
        self.config2 = Configuration(self.cs,
                                     values={'a': 100, 'b': 0})
        self.config3 = Configuration(self.cs,
                                     values={'a': 100, 'b': 100})
        self.config4 = Configuration(self.cs,
                                     values={'a': 0, 'b': 0})

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
            stats=self.stats,
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
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[1], initial_budget=1, max_budget=10, eta=2)

        self.assertEqual(len(intensifier.inst_seed_pairs), 1)  # since instance-seed pairs
        self.assertEqual(intensifier.initial_budget, 1)
        self.assertEqual(intensifier.max_budget, 10)
        self.assertListEqual(intensifier.n_configs_in_stage, [8.0, 4.0, 2.0, 1.0])
        self.assertListEqual(list(intensifier.all_budgets), [1.25, 2.5, 5., 10.])
        self.assertFalse(intensifier.instance_as_budget)
        self.assertFalse(intensifier.repeat_configs)

    def test_init_3(self):
        """
            test parameter initialiations for successive halving - real-valued budget, high initial budget
        """
        intensifier = SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[1], initial_budget=9, max_budget=10, eta=2)

        self.assertEqual(len(intensifier.inst_seed_pairs), 1)  # since instance-seed pairs
        self.assertEqual(intensifier.initial_budget, 9)
        self.assertEqual(intensifier.max_budget, 10)
        self.assertListEqual(intensifier.n_configs_in_stage, [1.0])
        self.assertListEqual(list(intensifier.all_budgets), [10.])
        self.assertFalse(intensifier.instance_as_budget)
        self.assertFalse(intensifier.repeat_configs)

    def test_init_4(self):
        """
            test wrong parameter initializations for successive halving
        """

        # runtime as budget (no param provided)
        with self.assertRaisesRegex(ValueError,
                                    "requires parameters initial_budget and max_budget for intensification!"):
            SuccessiveHalving(
                stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1])

        # eta < 1
        with self.assertRaisesRegex(ValueError, "eta must be greater than 1"):
            SuccessiveHalving(
                stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1], eta=0)

        # max budget > instance-seed pairs
        with self.assertRaisesRegex(ValueError,
                                    "Max budget cannot be greater than the number of instance-seed pairs"):
            SuccessiveHalving(
                stats=self.stats,
                traj_logger=TrajLogger(output_dir=None, stats=self.stats),
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                cutoff=10, instances=[1, 2, 3], initial_budget=1, max_budget=5, n_seeds=1)

    def test_top_k_1(self):
        """
            test _top_k() for configs with same instance-seed-budget keys
        """
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345),
            instances=[1, 2], initial_budget=1)
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
        self.rh.add(config=self.config3, cost=3, time=3,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config3, cost=3, time=3,
                    status=StatusType.SUCCESS, instance_id=2, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config4, cost=0.5, time=0.5,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config4, cost=0.5, time=0.5,
                    status=StatusType.SUCCESS, instance_id=2, seed=None,
                    additional_info=None)
        conf = intensifier._top_k(configs=[self.config1, self.config2, self.config3, self.config4],
                                  k=2, run_history=self.rh)

        # Check that config4 is also before config1 (as it has the lower cost)
        self.assertEqual(conf, [self.config4, self.config1])

    def test_top_k_2(self):
        """
            test _top_k() for configs with different instance-seed-budget keys
        """
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345),
            instances=[1, 2], initial_budget=1)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=10, time=10,
                    status=StatusType.SUCCESS, instance_id=2, seed=None,
                    additional_info=None)

        with self.assertRaisesRegex(ValueError, 'Cannot compare configs'):
            intensifier._top_k(configs=[self.config2, self.config1, self.config3],
                               k=1, run_history=self.rh)

    def test_top_k_3(self):
        """
            test _top_k() for not enough configs to generate for the next budget
        """
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345),
            instances=[1], initial_budget=1)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=1, time=1,
                    status=StatusType.CRASHED, instance_id=1, seed=None,
                    additional_info=None)
        configs = intensifier._top_k(configs=[self.config1], k=2, run_history=self.rh)

        # top_k should return whatever configuration is possible
        self.assertEqual(configs, [self.config1])

    def test_top_k_4(self):
        """
            test _top_k() for not enough configs to generate for the next budget
        """
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None, run_obj_time=False,
            rng=np.random.RandomState(12345), eta=2, num_initial_challengers=4,
            instances=[1], initial_budget=1, max_budget=10)
        intensifier._update_stage(self.rh)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None, budget=1,
                    additional_info=None)
        self.rh.add(config=self.config2, cost=1, time=1,
                    status=StatusType.DONOTADVANCE, instance_id=1, seed=None, budget=1,
                    additional_info=None)
        self.rh.add(config=self.config3, cost=1, time=1,
                    status=StatusType.DONOTADVANCE, instance_id=1, seed=None, budget=1,
                    additional_info=None)
        self.rh.add(config=self.config4, cost=1, time=1,
                    status=StatusType.DONOTADVANCE, instance_id=1, seed=None, budget=1,
                    additional_info=None)
        intensifier.success_challengers.add(self.config1)
        intensifier.fail_challengers.add(self.config2)
        intensifier.fail_challengers.add(self.config3)
        intensifier.fail_challengers.add(self.config4)
        intensifier._update_stage(self.rh)
        self.assertEqual(intensifier.fail_chal_offset, 1)  # we miss one challenger for this round
        configs = intensifier._top_k(configs=[self.config1], k=2, run_history=self.rh)
        self.assertEqual(configs, [self.config1])

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.DONOTADVANCE, instance_id=1, seed=None,
                    budget=intensifier.all_budgets[1], additional_info=None)
        intensifier.fail_challengers.add(self.config2)
        intensifier._update_stage(self.rh)
        self.assertEqual(intensifier.stage, 0)  # going back, since there are not enough to advance

    def test_get_next_run_1(self):
        """
            test get_next_run for a presently running configuration
        """
        def target(x):
            return 1

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')

        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[1, 2], initial_budget=1, max_budget=2, eta=2)

        # next challenger from a list
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )
        self.assertEqual(run_info.config, self.config1)
        self.assertTrue(intensifier.new_challenger)

        # until evaluated, does not pick new challenger
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )
        self.assertEqual(run_info.config, self.config1)
        self.assertEqual(intensifier.running_challenger, run_info.config)
        self.assertFalse(intensifier.new_challenger)

        # evaluating configuration
        self.assertIsNotNone(run_info.config)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
            log_traj=False,
        )
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        self.assertEqual(run_info.config, self.config2)
        self.assertEqual(len(intensifier.success_challengers), 1)
        self.assertTrue(intensifier.new_challenger)

    def test_get_next_run_2(self):
        """
            test get_next_run for higher stages of SH iteration
        """
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[1], initial_budget=1, max_budget=2, eta=2)

        intensifier._update_stage(run_history=None)
        intensifier.stage += 1
        intensifier.configs_to_run = [self.config1]

        # next challenger should come from configs to run
        intent, run_info = intensifier.get_next_run(
            challengers=None,
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )
        self.assertEqual(run_info.config, self.config1)
        self.assertEqual(len(intensifier.configs_to_run), 0)
        self.assertFalse(intensifier.new_challenger)

    def test_update_stage(self):
        """
            test update_stage - initializations for all tracking variables
        """
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[1], initial_budget=1, max_budget=2, eta=2)

        # first stage update
        intensifier._update_stage(run_history=None)

        self.assertEqual(intensifier.stage, 0)
        self.assertEqual(intensifier.sh_iters, 0)
        self.assertEqual(intensifier.running_challenger, None)
        self.assertEqual(intensifier.success_challengers, set())

        # higher stages
        self.rh.add(self.config1, 1, 1, StatusType.SUCCESS)
        self.rh.add(self.config2, 2, 2, StatusType.SUCCESS)
        intensifier.success_challengers = {self.config1, self.config2}
        intensifier._update_stage(run_history=self.rh)

        self.assertEqual(intensifier.stage, 1)
        self.assertEqual(intensifier.sh_iters, 0)
        self.assertEqual(intensifier.configs_to_run, [self.config1])

        # next iteration
        intensifier.success_challengers = {self.config1}
        intensifier._update_stage(run_history=self.rh)

        self.assertEqual(intensifier.stage, 0)
        self.assertEqual(intensifier.sh_iters, 1)
        self.assertIsInstance(intensifier.configs_to_run, list)
        self.assertEqual(len(intensifier.configs_to_run), 0)

    @unittest.mock.patch.object(SuccessiveHalving, '_top_k')
    def test_update_stage_2(self, top_k_mock):
        """
            test update_stage - everything good is in state do not advance
        """

        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, initial_budget=1, max_budget=4, eta=2, instances=None)

        # update variables
        intensifier._update_stage(run_history=None)

        intensifier.success_challengers.add(self.config1)
        intensifier.success_challengers.add(self.config2)
        intensifier.do_not_advance_challengers.add(self.config3)
        intensifier.do_not_advance_challengers.add(self.config4)

        top_k_mock.return_value = [self.config1, self.config3]

        # Test that we update the stage as there is one configuration advanced to the next budget
        self.assertEqual(intensifier.stage, 0)
        intensifier._update_stage(run_history=None)
        self.assertEqual(intensifier.stage, 1)
        self.assertEqual(intensifier.configs_to_run, [self.config1])
        self.assertEqual(intensifier.fail_chal_offset, 1)
        self.assertEqual(len(intensifier.success_challengers), 0)
        self.assertEqual(len(intensifier.do_not_advance_challengers), 0)

        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, initial_budget=1, max_budget=4, eta=2, instances=None)

        # update variables
        intensifier._update_stage(run_history=None)

        intensifier.success_challengers.add(self.config1)
        intensifier.success_challengers.add(self.config2)
        intensifier.do_not_advance_challengers.add(self.config3)
        intensifier.do_not_advance_challengers.add(self.config4)

        top_k_mock.return_value = [self.config3, self.config4]

        # Test that we update the stage as there is no configuration advanced to the next budget
        self.assertEqual(intensifier.stage, 0)
        intensifier._update_stage(run_history=None)
        self.assertEqual(intensifier.stage, 0)
        self.assertEqual(intensifier.configs_to_run, [])
        self.assertEqual(intensifier.fail_chal_offset, 0)
        self.assertEqual(len(intensifier.success_challengers), 0)
        self.assertEqual(len(intensifier.do_not_advance_challengers), 0)

        top_k_mock.return_value = []

    def test_eval_challenger_1(self):
        """
           test eval_challenger with quality objective & real-valued budget
        """

        def target(x: Configuration, instance: str, seed: int, budget: float):
            return 0.1 * budget

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            cutoff=1, instances=[None], initial_budget=0.25, max_budget=0.5, eta=2)
        intensifier._update_stage(run_history=None)

        self.rh.add(config=self.config1, cost=1, time=1, status=StatusType.SUCCESS,
                    seed=0, budget=0.5)
        self.rh.add(config=self.config2, cost=1, time=1, status=StatusType.SUCCESS,
                    seed=0, budget=0.25)
        self.rh.add(config=self.config3, cost=2, time=1, status=StatusType.SUCCESS,
                    seed=0, budget=0.25)

        intensifier.success_challengers = {self.config2, self.config3}
        intensifier._update_stage(run_history=self.rh)

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1],
            chooser=None,
            incumbent=self.config1,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(inc, self.config2)
        self.assertEqual(inc_value, 0.05)
        self.assertEqual(self.stats.ta_runs, 1)
        self.assertEqual(list(self.rh.data.keys())[-1][0], self.rh.config_ids[self.config2])
        self.assertEqual(self.stats.inc_changed, 1)

    def test_eval_challenger_2(self):
        """
           test eval_challenger with runtime objective and adaptive capping
        """

        def target(x: Configuration, instance: str):
            if x['a'] == 100 or instance == 2:
                time.sleep(1.5)
            return (x['a'] + 1) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime")
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, cutoff=1,
            instances=[1, 2], initial_budget=1, max_budget=2, eta=2, instance_order=None)

        # config1 should be executed successfully and selected as incumbent
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1],
            chooser=None,
            incumbent=None,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(run_info.config, self.config1)
        self.assertEqual(self.stats.ta_runs, 1)
        self.assertEqual(self.stats.inc_changed, 1)

        # config2 should be capped and config1 should still be the incumbent

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.ta_runs, 2)
        self.assertEqual(self.stats.inc_changed, 1)
        self.assertEqual(list(self.rh.data.values())[1][2], StatusType.CAPPED)

        # config1 is selected for the next stage and allowed to timeout since this is the 1st run for this instance
        intent, run_info = intensifier.get_next_run(
            challengers=[],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.ta_runs, 3)
        self.assertEqual(list(self.rh.data.values())[2][2], StatusType.TIMEOUT)

    @mock.patch.object(SuccessiveHalving, '_top_k')
    def test_eval_challenger_capping(self, patch):
        """
            test eval_challenger with adaptive capping and all configurations capped/crashed
        """

        def target(x):
            if x['b'] == 100:
                time.sleep(1.5)
            if x['a'] == 100:
                raise ValueError('You shall not pass')
            return (x['a'] + 1) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime",
                                abort_on_first_run_crash=False)
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, cutoff=1,
            instances=[1, 2], initial_budget=1, max_budget=2, eta=2, instance_order=None)

        for i in range(2):
            self.rh.add(config=self.config1, cost=.001, time=0.001,
                        status=StatusType.SUCCESS, instance_id=i + 1, seed=0,
                        additional_info=None)

        # provide configurations
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            incumbent=self.config1,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.ta_runs, 1)
        self.assertEqual(list(self.rh.data.values())[2][2], StatusType.CRASHED)
        self.assertEqual(len(intensifier.success_challengers), 0)

        # provide configurations
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config3],
            chooser=None,
            incumbent=self.config1,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config1)
        self.assertEqual(self.stats.ta_runs, 2)
        self.assertEqual(self.stats.inc_changed, 0)
        self.assertEqual(list(self.rh.data.values())[3][2], StatusType.CAPPED)
        self.assertEqual(len(intensifier.success_challengers), 0)
        # top_k() should not be called since all configs were capped
        self.assertFalse(patch.called)

        # now the SH iteration should begin a new iteration since all configs were capped!
        self.assertEqual(intensifier.sh_iters, 1)
        self.assertEqual(intensifier.stage, 0)

        # should raise an error as this is a new iteration but no configs were provided
        with self.assertRaisesRegex(ValueError, 'No configurations/chooser provided.'):
            intent, run_info = intensifier.get_next_run(
                challengers=None,
                chooser=None,
                incumbent=self.config1,
                run_history=self.rh
            )

    def test_eval_challenger_capping_2(self):
        """
            test eval_challenger for adaptive capping with all but one configuration capped
        """
        def target(x):
            if x['a'] + x['b'] > 0:
                time.sleep(1.5)
            return x['a']

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime")
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=False, cutoff=1,
            instances=[1, 2], n_seeds=2, initial_budget=1, max_budget=4, eta=2, instance_order=None)

        # first configuration run
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config4],
            chooser=None,
            incumbent=None,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(inc, self.config4)

        # remaining 3 runs should be capped
        for i in [self.config1, self.config2, self.config3]:
            intent, run_info = intensifier.get_next_run(
                challengers=[i],
                chooser=None,
                incumbent=inc,
                run_history=self.rh
            )
            result = eval_challenger(run_info, taf, self.stats, self.rh)
            inc, inc_value = intensifier.process_results(
                run_info=run_info,
                incumbent=inc,
                run_history=self.rh,
                time_bound=np.inf,
                result=result,
            )

        self.assertEqual(inc, self.config4)
        self.assertEqual(list(self.rh.data.values())[1][2], StatusType.CAPPED)
        self.assertEqual(list(self.rh.data.values())[2][2], StatusType.CAPPED)
        self.assertEqual(list(self.rh.data.values())[3][2], StatusType.CAPPED)
        self.assertEqual(intensifier.stage, 1)
        self.assertEqual(intensifier.fail_chal_offset, 1)  # 2 configs expected, but 1 failure

        # run next stage - should run only 1 configuration since other 3 were capped
        # 1 runs for config1
        intent, run_info = intensifier.get_next_run(
            challengers=[],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        self.assertEqual(run_info.config, self.config4)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(intensifier.stage, 2)

        # run next stage with only config1
        # should go to next iteration since no more configurations left
        for _ in range(2):
            intent, run_info = intensifier.get_next_run(
                challengers=[],
                chooser=None,
                incumbent=inc,
                run_history=self.rh
            )
            self.assertEqual(run_info.config, self.config4)
            result = eval_challenger(run_info, taf, self.stats, self.rh)
            inc, inc_value = intensifier.process_results(
                run_info=run_info,
                incumbent=inc,
                run_history=self.rh,
                time_bound=np.inf,
                result=result,
            )

        self.assertEqual(inc, self.config4)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config4, only_max_observed_budget=True)), 4)
        self.assertEqual(intensifier.sh_iters, 1)
        self.assertEqual(intensifier.stage, 0)

        with self.assertRaisesRegex(ValueError, 'No configurations/chooser provided.'):
            intensifier.get_next_run(
                challengers=[],
                chooser=None,
                incumbent=inc,
                run_history=self.rh
            )

    def test_eval_challenger_3(self):
        """
            test eval_challenger for updating to next stage and shuffling instance order every run
        """

        def target(x: Configuration, instance: str):
            return (x['a'] + int(instance)) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="quality")
        taf.runhistory = self.rh

        intensifier = SuccessiveHalving(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=[0, 1], instance_order='shuffle', eta=2,
            deterministic=True, cutoff=1)

        intensifier._update_stage(run_history=self.rh)

        self.assertEqual(intensifier.inst_seed_pairs, [(0, 0), (1, 0)])

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1],
            chooser=None,
            incumbent=None,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(inc, self.config1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config1, only_max_observed_budget=True)), 1)
        self.assertEqual(intensifier.configs_to_run, [])
        self.assertEqual(intensifier.stage, 0)

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(inc, self.config1)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2, only_max_observed_budget=True)), 1)
        self.assertEqual(intensifier.configs_to_run, [self.config1])  # Incumbent is promoted to the next stage
        self.assertEqual(intensifier.stage, 1)

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config3],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(inc, self.config1)

        self.assertEqual(len(self.rh.get_runs_for_config(self.config1, only_max_observed_budget=True)), 2)
        self.assertEqual(intensifier.sh_iters, 1)
        self.assertEqual(self.stats.inc_changed, 1)

        # For the 2nd SH iteration, we should still be able to run the old configurations again
        # since instance order is "shuffle"

        self.assertEqual(intensifier.inst_seed_pairs, [(1, 0), (0, 0)])

        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2],
            chooser=None,
            incumbent=inc,
            run_history=self.rh
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=inc,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(run_info.config, self.config2)
        self.assertEqual(len(self.rh.get_runs_for_config(self.config2, only_max_observed_budget=True)), 2)

    def test_incumbent_selection_default(self):
        """
            test _compare_config for default incumbent selection design (highest budget so far)
        """
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=[1], initial_budget=1, max_budget=2, eta=2)
        intensifier.stage = 0

        # SH considers challenger as incumbent in first run in eval_challenger
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=1)
        inc = intensifier._compare_configs(challenger=self.config1, incumbent=self.config1,
                                           run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config1)
        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=2)
        inc = intensifier._compare_configs(challenger=self.config1, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config1)

        # Adding a worse configuration
        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=1)
        inc = intensifier._compare_configs(challenger=self.config2, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config1)
        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=2)
        inc = intensifier._compare_configs(challenger=self.config2, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config1)

        # Adding a better configuration, but the incumbent will only be changed on budget=2
        self.rh.add(config=self.config3, cost=0.5, time=3,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=1)
        inc = intensifier._compare_configs(challenger=self.config3, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config1)
        self.rh.add(config=self.config3, cost=0.5, time=3,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=2)
        inc = intensifier._compare_configs(challenger=self.config3, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config3)

        # Test that the state is only based on the runhistory
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345),
            instances=[1], initial_budget=1)
        intensifier.stage = 0

        # Adding a better configuration, but the incumbent will only be changed on budget=2
        self.rh.add(config=self.config4, cost=0.1, time=3,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=1)
        inc = intensifier._compare_configs(challenger=self.config4, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config3)
        self.rh.add(config=self.config4, cost=0.1, time=3,
                    status=StatusType.SUCCESS, instance_id=1, seed=None,
                    additional_info=None, budget=2)
        inc = intensifier._compare_configs(challenger=self.config4, incumbent=inc, run_history=self.rh, log_traj=False)
        self.assertEqual(inc, self.config4)

    def test_incumbent_selection_designs(self):
        """
            test _compare_config with different incumbent selection designs
        """

        # select best on any budget
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=[1], initial_budget=1, max_budget=2, eta=2, incumbent_selection='any_budget')
        intensifier.stage = 0

        self.rh.add(config=self.config1, instance_id=1, seed=None, budget=1,
                    cost=0.5, time=1, status=StatusType.SUCCESS, additional_info=None)
        self.rh.add(config=self.config1, instance_id=1, seed=None, budget=2,
                    cost=10, time=1, status=StatusType.SUCCESS, additional_info=None)
        self.rh.add(config=self.config2, instance_id=1, seed=None, budget=2,
                    cost=5, time=1, status=StatusType.SUCCESS, additional_info=None)

        # incumbent should be config1, since it has the best performance in one of the budgets
        inc = intensifier._compare_configs(incumbent=self.config2, challenger=self.config1,
                                           run_history=self.rh, log_traj=False)
        self.assertEqual(self.config1, inc)
        # if config1 is incumbent already, it shouldn't change
        inc = intensifier._compare_configs(incumbent=self.config1, challenger=self.config2,
                                           run_history=self.rh, log_traj=False)
        self.assertEqual(self.config1, inc)

        # select best on highest budget only
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=[1], initial_budget=1, max_budget=4, eta=2, incumbent_selection='highest_budget')
        intensifier.stage = 0

        # incumbent should not change, since there is no run on the highest budget,
        # though config3 is run on a higher budget
        self.rh.add(config=self.config3, instance_id=1, seed=None, budget=2,
                    cost=0.5, time=1, status=StatusType.SUCCESS, additional_info=None)
        self.rh.add(config=self.config4, instance_id=1, seed=None, budget=1,
                    cost=5, time=1, status=StatusType.SUCCESS, additional_info=None)
        inc = intensifier._compare_configs(incumbent=self.config4, challenger=self.config3,
                                           run_history=self.rh, log_traj=False)
        self.assertEqual(self.config4, inc)
        self.assertEqual(self.stats.inc_changed, 0)

        # incumbent changes to config3 since that is run on the highest budget
        self.rh.add(config=self.config3, instance_id=1, seed=None, budget=4,
                    cost=10, time=1, status=StatusType.SUCCESS, additional_info=None)
        inc = intensifier._compare_configs(incumbent=self.config4, challenger=self.config3,
                                           run_history=self.rh, log_traj=False)
        self.assertEqual(self.config3, inc)

    def test_count_running_instances_for_challenger(self):
        """
        Makes sure we return the proper number of running configs
        """
        def target(x):
            return 1
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')

        taf.runhistory = self.rh
        # select best on any budget
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=list(range(20)), initial_budget=3, max_budget=10, eta=2)

        # Get and process the first instance pair of config 1
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config1, self.config2, self.config3, self.config4],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
            log_traj=False,
        )
        self.assertEqual(intensifier._count_running_instances_for_challenger(self.rh), 0)

        # This will get us the second instance of config 1
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2, self.config3, self.config4],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )

        # Mark this run_info object as running and make sure that the intensifier can
        # access it
        self.rh.add(config=run_info.config,
                    instance_id=run_info.instance,
                    seed=run_info.seed,
                    budget=run_info.budget,
                    cost=10,
                    time=1,
                    status=StatusType.RUNNING,
                    additional_info=None)
        self.assertEqual(intensifier._count_running_instances_for_challenger(self.rh), 1)

        # Now, when the intensifier is requested, to provide a new run,
        # it should provide a new instance, not the same one
        new_intent, new_run_info = intensifier.get_next_run(
            challengers=[self.config2, self.config3, self.config4],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )
        self.assertNotEqual(new_run_info, run_info)

        # Add another running instance
        self.rh.add(config=new_run_info.config,
                    instance_id=new_run_info.instance,
                    seed=new_run_info.seed,
                    budget=new_run_info.budget,
                    cost=10,
                    time=1,
                    status=StatusType.RUNNING,
                    additional_info=None)
        self.assertEqual(intensifier._count_running_instances_for_challenger(self.rh), 2)

        # Get a new config to gain more confidence
        newer_intent, newer_run_info = intensifier.get_next_run(
            challengers=[self.config2, self.config3, self.config4],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )
        self.assertNotIn(newer_run_info, [new_run_info, run_info])

    def test_launched_all_configs_for_current_stage(self):
        """
        Makes sure we return the proper number of running configs
        """
        def target(x):
            return 1
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')

        taf.runhistory = self.rh
        # select best on any budget
        intensifier = SuccessiveHalving(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), run_obj_time=False,
            instances=list(range(10)), initial_budget=2, max_budget=10, eta=2)

        # So there are 2 instances per config.
        # self.stage=0
        # self.n_configs_in_stage=[4.0, 2.0, 1.0]
        # self.all_budgets=[ 2.5  5.  10. ]
        total_configs_in_stage = 4
        instances_per_stage = 2

        # get all configs and add them to list
        run_tracker = []
        challengers = [self.config1, self.config2, self.config3, self.config4]
        for i in range(total_configs_in_stage * instances_per_stage):
            intent, run_info = intensifier.get_next_run(
                challengers=challengers,
                chooser=None,
                run_history=self.rh,
                incumbent=None,
            )

            # All this runs are valid for this stage
            self.assertEqual(intent, RunInfoIntent.RUN)

            # Remove from the challengers, the launched configs
            challengers = [c for c in challengers if c != run_info.config]
            run_tracker.append((run_info.config, run_info.instance, run_info.seed))
            self.rh.add(config=run_info.config,
                        instance_id=run_info.instance,
                        seed=run_info.seed,
                        budget=run_info.budget,
                        cost=10,
                        time=1,
                        status=StatusType.RUNNING,
                        additional_info=None)
        print(intensifier._launched_all_configs_for_current_stage(self.rh))

        # This will get us the second instance of config 1
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2, self.config3, self.config4],
            chooser=None,
            run_history=self.rh,
            incumbent=None,
        )

        # We have launched all runs, that are expected for this stage
        # not registered any, so for sure we have to wait
        # For all runs to be completed before moving to the next stage
        self.assertEqual(intent, RunInfoIntent.WAIT)
