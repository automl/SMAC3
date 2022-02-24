import unittest
from unittest import mock

import logging
import numpy as np
import time

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.intensification.abstract_racer import RunInfoIntent
from smac.intensification.successive_halving import _SuccessiveHalving
from smac.intensification.hyperband import _Hyperband
from smac.intensification.hyperband import Hyperband
from smac.runhistory.runhistory import RunHistory, RunInfo, RunValue
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict
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


def target_from_run_info(RunInfo):
    value_from_config = sum([a for a in RunInfo.config.get_dictionary().values()])
    return RunValue(
        cost=value_from_config,
        time=0.5,
        status=StatusType.SUCCESS,
        starttime=time.time(),
        endtime=time.time() + 1,
        additional_info={}
    )


class TestHyperband(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.rh = RunHistory()
        self.cs = get_config_space()
        self.config1 = Configuration(self.cs,
                                     values={'a': 7, 'b': 11})
        self.config2 = Configuration(self.cs,
                                     values={'a': 13, 'b': 17})
        self.config3 = Configuration(self.cs,
                                     values={'a': 0, 'b': 7})
        self.config4 = Configuration(self.cs,
                                     values={'a': 29, 'b': 31})
        self.config5 = Configuration(self.cs,
                                     values={'a': 31, 'b': 33})

        self.scen = Scenario({"cutoff_time": 2, 'cs': self.cs,
                              "run_obj": 'runtime',
                              "output_dir": ''})
        self.stats = Stats(scenario=self.scen)
        self.stats.start_timing()

        # Create the base object
        self.HB = Hyperband(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            deterministic=False,
            run_obj_time=False,
            instances=[1, 2, 3, 4, 5],
            n_seeds=2,
            initial_budget=2,
            max_budget=5,
            eta=2,
        )

    def test_initialization(self):
        """Makes sure that a proper_HB is created"""

        # We initialize the HB with zero intensifier_instances
        self.assertEqual(len(self.HB.intensifier_instances), 0)

        # Add an instance to check the_HB initialization
        self.assertTrue(self.HB._add_new_instance(num_workers=1))

        # Some default init
        self.assertEqual(self.HB.intensifier_instances[0].hb_iters, 0)
        self.assertEqual(self.HB.intensifier_instances[0].max_budget, 5)
        self.assertEqual(self.HB.intensifier_instances[0].initial_budget, 2)

        # Update the stage
        (self.HB.intensifier_instances[0]._update_stage(self.rh))

        # Parameters properly passed to SH
        self.assertEqual(len(self.HB.intensifier_instances[0].sh_intensifier.inst_seed_pairs), 10)
        self.assertEqual(self.HB.intensifier_instances[0].sh_intensifier.initial_budget, 2)
        self.assertEqual(self.HB.intensifier_instances[0].sh_intensifier.max_budget, 5)

    def test_process_results_via_sourceid(self):
        """Makes sure source id is honored when deciding
        which_HB instance will consume the result/run_info

        """
        # Mock the_HB instance so we can make sure the correct item is passed
        for i in range(10):
            self.HB.intensifier_instances[i] = mock.Mock()
            self.HB.intensifier_instances[i].process_results.return_value = (self.config1, 0.5)
            # make iter false so the mock object is not overwritten
            self.HB.intensifier_instances[i].iteration_done = False

        # randomly create run_infos and push into HB. Then we will make
        # sure they got properly allocated
        for i in np.random.choice(list(range(10)), 30):
            run_info = RunInfo(
                config=self.config1,
                instance=0,
                instance_specific="0",
                cutoff=None,
                seed=0,
                capped=False,
                budget=0.0,
                source_id=i,
            )

            # make sure results aren't messed up via magic variable
            # That is we check only the proper_HB instance has this
            magic = time.time()

            result = RunValue(
                cost=1,
                time=0.5,
                status=StatusType.SUCCESS,
                starttime=1,
                endtime=2,
                additional_info=magic
            )
            self.HB.process_results(
                run_info=run_info,
                incumbent=None,
                run_history=self.rh,
                time_bound=None,
                result=result,
                log_traj=False
            )

            # Check the call arguments of each sh instance and make sure
            # it is the correct one

            # First the expected one
            self.assertEqual(
                self.HB.intensifier_instances[i].process_results.call_args[1]['run_info'], run_info)
            self.assertEqual(
                self.HB.intensifier_instances[i].process_results.call_args[1]['result'], result)
            all_other_run_infos, all_other_results = [], []
            for j in range(len(self.HB.intensifier_instances)):
                # Skip the expected_HB instance
                if i == j:
                    continue
                if self.HB.intensifier_instances[j].process_results.call_args is None:
                    all_other_run_infos.append(None)
                else:
                    all_other_run_infos.append(
                        self.HB.intensifier_instances[j].process_results.call_args[1]['run_info'])
                    all_other_results.append(
                        self.HB.intensifier_instances[j].process_results.call_args[1]['result'])
            self.assertNotIn(run_info, all_other_run_infos)
            self.assertNotIn(result, all_other_results)

    def test_get_next_run_single_HB_instance(self):
        """Makes sure that a single_HB instance returns a valid config"""

        challengers = [self.config1, self.config2, self.config3, self.config4]
        for i in range(30):
            intent, run_info = self.HB.get_next_run(
                challengers=challengers,
                incumbent=None, chooser=None, run_history=self.rh,
                num_workers=1,
            )

            # Regenerate challenger list
            challengers = [c for c in challengers if c != run_info.config]

            if intent == RunInfoIntent.WAIT:
                break

            # Add the config to self.rh in order to make HB aware that this
            # config/instance was launched
            self.rh.add(
                config=run_info.config,
                cost=10,
                time=0.0,
                status=StatusType.RUNNING,
                instance_id=run_info.instance,
                seed=run_info.seed,
                budget=run_info.budget,
            )

        # smax==1 (int(np.floor(np.log(self.max_budget / self.initial_budget) / np.log(self.   eta))))
        self.assertEqual(self.HB.intensifier_instances[0].s_max, 1)

        # And we do not even complete 1 iteration, so s has to be 1
        self.assertEqual(self.HB.intensifier_instances[0].s, 1)

        # We should not create more_HB instance intensifier_instances
        self.assertEqual(len(self.HB.intensifier_instances), 1)

        # We are running with:
        # 'all_budgets': array([2.5, 5. ]) -> 2 intensifier_instances per config top
        # 'n_configs_in_stage': [2.0, 1.0],
        # This means we run int(2.5) + 2.0 = 4 runs before waiting
        self.assertEqual(i, 4)

        # Also, check the internals of the unique sh instance

        # sh_initial_budget==2.5 (self.eta ** -self.s * self.max_budget)
        self.assertEqual(self.HB.intensifier_instances[0].sh_intensifier.initial_budget, 2)

        # n_challengers=2 (int(np.floor((self.s_max + 1) / (self.s + 1)) * self.eta ** self.s))
        self.assertEqual(len(self.HB.intensifier_instances[0].sh_intensifier.n_configs_in_stage), 2)

    def test_get_next_run_multiple_HB_instances(self):
        """Makes sure that two _HB instance can properly coexist and tag
        run_info properly"""

        # We allow 2_HB instance to be created. This means, we have a newer iteration
        # to expect in hyperband
        challengers = [self.config1, self.config2, self.config3, self.config4]
        run_infos = []
        for i in range(30):
            intent, run_info = self.HB.get_next_run(
                challengers=challengers,
                incumbent=None, chooser=None, run_history=self.rh,
                num_workers=2,
            )
            run_infos.append(run_info)

            # Regenerate challenger list
            challengers = [c for c in challengers if c != run_info.config]

            # Add the config to self.rh in order to make HB aware that this
            # config/instance was launched
            if intent == RunInfoIntent.WAIT:
                break
            self.rh.add(
                config=run_info.config,
                cost=10,
                time=0.0,
                status=StatusType.RUNNING,
                instance_id=run_info.instance,
                seed=run_info.seed,
                budget=run_info.budget,
            )

        # We have not completed an iteration
        self.assertEqual(self.HB.intensifier_instances[0].hb_iters, 0)

        # Because n workers is now 2, we expect 2 sh intensifier_instances
        self.assertEqual(len(self.HB.intensifier_instances), 2)

        # Each of the intensifier_instances should have s equal to 1
        # As no iteration has been completed
        self.assertEqual(self.HB.intensifier_instances[0].s_max, 1)
        self.assertEqual(self.HB.intensifier_instances[0].s, 1)
        self.assertEqual(self.HB.intensifier_instances[1].s_max, 1)
        self.assertEqual(self.HB.intensifier_instances[1].s, 1)

        # First let us check everything makes sense in_HB-SH-0 HB-SH-0
        self.assertEqual(self.HB.intensifier_instances[0].sh_intensifier.initial_budget, 2)
        self.assertEqual(self.HB.intensifier_instances[0].sh_intensifier.max_budget, 5)
        self.assertEqual(len(self.HB.intensifier_instances[0].sh_intensifier.n_configs_in_stage), 2)
        self.assertEqual(self.HB.intensifier_instances[1].sh_intensifier.initial_budget, 2)
        self.assertEqual(self.HB.intensifier_instances[1].sh_intensifier.max_budget, 5)
        self.assertEqual(len(self.HB.intensifier_instances[1].sh_intensifier.n_configs_in_stage), 2)

        # We are running with:
        # + 4 runs for sh instance 0 ('all_budgets': array([2.5, 5. ]), 'n_configs_in_stage': [2.0, 1.0])
        #   that is, for SH0 we run in stage==0 int(2.5) intensifier_instances * 2.0 configs
        # And this times 2 because we have 2_HB intensifier_instances
        self.assertEqual(i, 8)

        # Adding a new worker is not possible as we already have 2 intensifier_instances
        # and n_workers==2
        intent, run_info = self.HB.get_next_run(
            challengers=challengers,
            incumbent=None, chooser=None, run_history=self.rh,
            num_workers=2,
        )
        self.assertEqual(intent, RunInfoIntent.WAIT)

    def test_add_new_instance(self):
        """Test whether we can add a instance and when we should not"""

        # By default we do not create a_HB
        # test adding the first instance!
        self.assertEqual(len(self.HB.intensifier_instances), 0)
        self.assertTrue(self.HB._add_new_instance(num_workers=1))
        self.assertEqual(len(self.HB.intensifier_instances), 1)
        self.assertIsInstance(self.HB.intensifier_instances[0], _Hyperband)
        # A second call should not add a new_HB instance
        self.assertFalse(self.HB._add_new_instance(num_workers=1))

        # We try with 2_HB instance active

        # We effectively return true because we added a new_HB instance
        self.assertTrue(self.HB._add_new_instance(num_workers=2))

        self.assertEqual(len(self.HB.intensifier_instances), 2)
        self.assertIsInstance(self.HB.intensifier_instances[1], _Hyperband)

        # Trying to add a third one should return false
        self.assertFalse(self.HB._add_new_instance(num_workers=2))
        self.assertEqual(len(self.HB.intensifier_instances), 2)

    def _exhaust_run_and_get_incumbent(self, sh, rh, num_workers=2):
        """
        Runs all provided configs on all intensifier_instances and return the incumbent
        as a nice side effect runhistory/stats are properly filled
        """
        challengers = [self.config1, self.config2, self.config3, self.config4]
        incumbent = None
        for i in range(100):
            try:
                intent, run_info = sh.get_next_run(
                    challengers=challengers,
                    incumbent=None, chooser=None, run_history=rh,
                    num_workers=num_workers,
                )
            except ValueError as e:
                # Get configurations until you run out of them
                print(e)
                break

            # Regenerate challenger list
            challengers = [c for c in challengers if c != run_info.config]

            if intent == RunInfoIntent.WAIT:
                break

            result = target_from_run_info(run_info)
            rh.add(
                config=run_info.config,
                cost=result.cost,
                time=result.time,
                status=result.status,
                instance_id=run_info.instance,
                seed=run_info.seed,
                budget=run_info.budget,
            )
            incumbent, inc_perf = sh.process_results(
                run_info=run_info,
                incumbent=incumbent,
                run_history=rh,
                time_bound=100.0,
                result=result,
                log_traj=False,
            )
        return incumbent, inc_perf

    def test_parallel_same_as_serial_HB(self):
        """Makes sure we behave the same as a serial run at the end"""

        # Get the run_history for a_HB instance run:
        rh = RunHistory()
        stats = Stats(scenario=self.scen)
        stats.start_timing()
        _HB = _Hyperband(
            stats=stats,
            traj_logger=TrajLogger(output_dir=None, stats=stats),
            rng=np.random.RandomState(12345),
            deterministic=True,
            run_obj_time=False,
            instances=[1, 2, 3, 4, 5],
            initial_budget=2,
            max_budget=5,
            eta=2,
        )
        incumbent, inc_perf = self._exhaust_run_and_get_incumbent(_HB, rh, num_workers=1)

        # Just to make sure nothing has changed from the_HB instance side to make
        # this check invalid:
        # We add config values, so config 3 with 0 and 7 should be the lesser cost
        self.assertEqual(incumbent, self.config3)
        self.assertEqual(inc_perf, 7.0)

        # Do the same for HB, but have multiple_HB instance in there
        # This_HB instance will be created via num_workers==2
        # in self._exhaust_run_and_get_incumbent
        HB = Hyperband(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            deterministic=True,
            run_obj_time=False,
            instances=[1, 2, 3, 4, 5],
            initial_budget=2,
            max_budget=5,
            eta=2,
        )
        incumbent_phb, inc_perf_phb = self._exhaust_run_and_get_incumbent(HB, self.rh)
        self.assertEqual(incumbent, incumbent_phb)

        # This makes sure there is a single incumbent in HB
        self.assertEqual(inc_perf, inc_perf_phb)

        # We don't want to loose any configuration, and particularly
        # we want to make sure the values of_HB instance to HB match
        self.assertEqual(len(self.rh.data), len(rh.data))

        # Because it is a deterministic run, the run histories must be the
        # same on exhaustion
        self.assertDictEqual(self.rh.data, rh.data)


class Test_Hyperband(unittest.TestCase):

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

    def test_update_stage(self):
        """
            test initialization of all parameters and tracking variables
        """
        intensifier = _Hyperband(
            stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[1], initial_budget=0.1, max_budget=1, eta=2)
        intensifier._update_stage()

        self.assertEqual(intensifier.s, 3)
        self.assertEqual(intensifier.s_max, 3)
        self.assertEqual(intensifier.hb_iters, 0)
        self.assertIsInstance(intensifier.sh_intensifier, _SuccessiveHalving)
        self.assertEqual(intensifier.sh_intensifier.initial_budget, 0.125)
        self.assertEqual(intensifier.sh_intensifier.n_configs_in_stage, [8.0, 4.0, 2.0, 1.0])

        # next HB stage
        intensifier._update_stage()

        self.assertEqual(intensifier.s, 2)
        self.assertEqual(intensifier.hb_iters, 0)
        self.assertEqual(intensifier.sh_intensifier.initial_budget, 0.25)
        self.assertEqual(intensifier.sh_intensifier.n_configs_in_stage, [4.0, 2.0, 1.0])

        intensifier._update_stage()  # s = 1
        intensifier._update_stage()  # s = 0
        # HB iteration completed
        intensifier._update_stage()

        self.assertEqual(intensifier.s, intensifier.s_max)
        self.assertEqual(intensifier.hb_iters, 1)
        self.assertEqual(intensifier.sh_intensifier.initial_budget, 0.125)
        self.assertEqual(intensifier.sh_intensifier.n_configs_in_stage, [8.0, 4.0, 2.0, 1.0])

    def test_eval_challenger(self):
        """
            since hyperband uses eval_challenger and get_next_run of the internal successive halving,
            we don't test these method extensively
        """

        def target(x):
            return 0.1

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = _Hyperband(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[None], initial_budget=0.5, max_budget=1, eta=2)

        self.assertFalse(hasattr(intensifier, 's'))

        # Testing get_next_run - get next configuration
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2, self.config3],
            chooser=None,
            incumbent=None,
            run_history=self.rh)
        self.assertEqual(intensifier.s, intensifier.s_max)
        self.assertEqual(run_info.config, self.config2)

        # update to the last SH iteration of the given HB stage
        self.assertEqual(intensifier.s, 1)
        self.assertEqual(intensifier.s_max, 1)

        # We assume now that process results was called with below successes.
        # We track closely run execution through run_tracker, so this also
        # has to be update -- the fact that the succesive halving inside hyperband
        # processed the given configurations
        self.rh.add(config=self.config1, cost=1, time=1, status=StatusType.SUCCESS,
                    seed=0, budget=1)
        intensifier.sh_intensifier.run_tracker[(self.config1, None, 0, 1)] = True
        self.rh.add(config=self.config2, cost=2, time=2, status=StatusType.SUCCESS,
                    seed=0, budget=0.5)
        intensifier.sh_intensifier.run_tracker[(self.config2, None, 0, 0.5)] = True
        self.rh.add(config=self.config3, cost=3, time=2, status=StatusType.SUCCESS,
                    seed=0, budget=0.5)
        intensifier.sh_intensifier.run_tracker[(self.config3, None, 0, 0.5)] = True

        intensifier.sh_intensifier.success_challengers = {self.config2, self.config3}
        intensifier.sh_intensifier._update_stage(self.rh)
        intent, run_info = intensifier.get_next_run(
            challengers=[self.config2, self.config3],
            chooser=None,
            incumbent=None,
            run_history=self.rh)

        # evaluation should change the incumbent to config2
        self.assertIsNotNone(run_info.config)
        result = eval_challenger(run_info, taf, self.stats, self.rh)
        inc, inc_value = intensifier.process_results(
            run_info=run_info,
            incumbent=self.config1,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )

        self.assertEqual(inc, self.config2)
        self.assertEqual(intensifier.s, 0)
        self.assertEqual(inc_value, 0.1)
        self.assertEqual(list(self.rh.data.keys())[-1][0], self.rh.config_ids[self.config2])
        self.assertEqual(self.stats.inc_changed, 1)


class Test__Hyperband(unittest.TestCase):

    def test_budget_initialization(self):
        """
            Check computing budgets (only for non-instance cases)
        """
        intensifier = _Hyperband(
            stats=None, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=None, initial_budget=1, max_budget=81, eta=3
        )
        self.assertListEqual([1, 3, 9, 27, 81], intensifier.all_budgets.tolist())
        self.assertListEqual([81, 27, 9, 3, 1], intensifier.n_configs_in_stage)

        to_check = [
            # minb, maxb, eta, n_configs_in_stage, all_budgets
            [1, 81, 3, [81, 27, 9, 3, 1], [1, 3, 9, 27, 81]],
            [1, 600, 3, [243, 81, 27, 9, 3, 1],
             [2.469135, 7.407407, 22.222222, 66.666666, 200, 600]],
            [1, 100, 10, [100, 10, 1], [1, 10, 100]],
            [0.001, 1, 3, [729, 243, 81, 27, 9, 3, 1],
             [0.001371, 0.004115, 0.012345, 0.037037, 0.111111, 0.333333, 1.0]],
            [1, 1000, 3, [729, 243, 81, 27, 9, 3, 1],
             [1.371742, 4.115226, 12.345679, 37.037037, 111.111111, 333.333333, 1000.0]],
            [0.001, 100, 10, [100000, 10000, 1000, 100, 10, 1],
             [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]],
        ]

        for minb, maxb, eta, n_configs_in_stage, all_budgets in to_check:
            intensifier = _Hyperband(
                stats=None, traj_logger=None,
                rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
                instances=None, initial_budget=minb, max_budget=maxb, eta=eta
            )
            intensifier._init_sh_params(initial_budget=minb,
                                        max_budget=maxb,
                                        eta=eta,
                                        _all_budgets=None,
                                        _n_configs_in_stage=None,
                                        )
            for i in range(len(all_budgets) + 10):
                intensifier._update_stage()
                comp_budgets = intensifier.sh_intensifier.all_budgets.tolist()
                comp_configs = intensifier.sh_intensifier.n_configs_in_stage

                self.assertIsInstance(comp_configs, list)
                for c in comp_configs:
                    self.assertIsInstance(c, int)

                # all_budgets for SH is always a subset of all_budgets of HB
                np.testing.assert_array_almost_equal(all_budgets[i % len(all_budgets):],
                                                     comp_budgets, decimal=5)

                # The content of these lists might differ
                self.assertEqual(len(n_configs_in_stage[i % len(n_configs_in_stage):]),
                                 len(comp_configs))


if __name__ == "__main__":
    unittest.main()
