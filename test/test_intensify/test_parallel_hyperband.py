import unittest
from unittest import mock

import numpy as np
import time

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.runhistory.runhistory import RunHistory, RunInfo, RunValue
from smac.scenario.scenario import Scenario
from smac.intensification.abstract_racer import RunInfoIntent
from smac.intensification.successive_halving import SuccessiveHalving
from smac.intensification.hyperband import Hyperband
from smac.intensification.parallel_hyperband import ParallelHyperband
from smac.tae import StatusType
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


class TestParallelHyperband(unittest.TestCase):

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
        self.PHB = ParallelHyperband(
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
        """Makes sure that a proper HB is created"""

        # We initialize the PHB with zero intensifier_instances
        self.assertEqual(len(self.PHB.intensifier_instances), 0)

        # Add an instance to check the HB initialization
        self.assertTrue(self.PHB._add_new_instance(num_workers=1))

        # Some default init
        self.assertEqual(self.PHB.intensifier_instances[0].hb_iters, 0)
        self.assertEqual(self.PHB.intensifier_instances[0].max_budget, 5)
        self.assertEqual(self.PHB.intensifier_instances[0].initial_budget, 2)

        # Update the stage
        (self.PHB.intensifier_instances[0]._update_stage(self.rh))

        # Parameters properly passed to SH
        self.assertEqual(len(self.PHB.intensifier_instances[0].sh_intensifier.inst_seed_pairs), 10)
        self.assertEqual(self.PHB.intensifier_instances[0].sh_intensifier.initial_budget, 2)
        self.assertEqual(self.PHB.intensifier_instances[0].sh_intensifier.max_budget, 5)

    def test_process_results_via_sourceid(self):
        """Makes sure source id is honored when deciding
        which HB instance will consume the result/run_info

        """
        # Mock the HB instance so we can make sure the correct item is passed
        for i in range(10):
            self.PHB.intensifier_instances[i] = mock.Mock()
            self.PHB.intensifier_instances[i].process_results.return_value = (self.config1, 0.5)
            # make iter false so the mock object is not overwritten
            self.PHB.intensifier_instances[i].iteration_done = False

        # randomly create run_infos and push into PHB. Then we will make
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
            # That is we check only the proper HB instance has this
            magic = time.time()

            result = RunValue(
                cost=1,
                time=0.5,
                status=StatusType.SUCCESS,
                starttime=1,
                endtime=2,
                additional_info=magic
            )
            self.PHB.process_results(
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
                self.PHB.intensifier_instances[i].process_results.call_args[1]['run_info'], run_info)
            self.assertEqual(
                self.PHB.intensifier_instances[i].process_results.call_args[1]['result'], result)
            all_other_run_infos, all_other_results = [], []
            for j in range(len(self.PHB.intensifier_instances)):
                # Skip the expected HB instance
                if i == j:
                    continue
                if self.PHB.intensifier_instances[j].process_results.call_args is None:
                    all_other_run_infos.append(None)
                else:
                    all_other_run_infos.append(
                        self.PHB.intensifier_instances[j].process_results.call_args[1]['run_info'])
                    all_other_results.append(
                        self.PHB.intensifier_instances[j].process_results.call_args[1]['result'])
            self.assertNotIn(run_info, all_other_run_infos)
            self.assertNotIn(result, all_other_results)

    def test_get_next_run_single_HB_instance(self):
        """Makes sure that a single HB instance returns a valid config"""

        challengers = [self.config1, self.config2, self.config3, self.config4]
        for i in range(30):
            intent, run_info = self.PHB.get_next_run(
                challengers=challengers,
                incumbent=None, chooser=None, run_history=self.rh,
                num_workers=1,
            )

            # Regenerate challenger list
            challengers = [c for c in challengers if c != run_info.config]

            if intent == RunInfoIntent.WAIT:
                break

            # Add the config to self.rh in order to make PHB aware that this
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
        self.assertEqual(self.PHB.intensifier_instances[0].s_max, 1)

        # And we do not even complete 1 iteration, so s has to be 1
        self.assertEqual(self.PHB.intensifier_instances[0].s, 1)

        # We should not create more HB instance intensifier_instances
        self.assertEqual(len(self.PHB.intensifier_instances), 1)

        # We are running with:
        # 'all_budgets': array([2.5, 5. ]) -> 2 intensifier_instances per config top
        # 'n_configs_in_stage': [2.0, 1.0],
        # This means we run int(2.5) + 2.0 = 4 runs before waiting
        self.assertEqual(i, 4)

        # Also, check the internals of the unique sh instance

        # sh_initial_budget==2.5 (self.eta ** -self.s * self.max_budget)
        self.assertEqual(self.PHB.intensifier_instances[0].sh_intensifier.initial_budget, 2)

        # n_challengers=2 (int(np.floor((self.s_max + 1) / (self.s + 1)) * self.eta ** self.s))
        self.assertEqual(len(self.PHB.intensifier_instances[0].sh_intensifier.n_configs_in_stage), 2)

    def test_get_next_run_multiple_HB_instances(self):
        """Makes sure that two  HB instance can properly coexist and tag
        run_info properly"""

        # We allow 2 HB instance to be created. This means, we have a newer iteration
        # to expect in hyperband
        challengers = [self.config1, self.config2, self.config3, self.config4]
        run_infos = []
        for i in range(30):
            intent, run_info = self.PHB.get_next_run(
                challengers=challengers,
                incumbent=None, chooser=None, run_history=self.rh,
                num_workers=2,
            )
            run_infos.append(run_info)

            # Regenerate challenger list
            challengers = [c for c in challengers if c != run_info.config]

            # Add the config to self.rh in order to make PHB aware that this
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
        self.assertEqual(self.PHB.intensifier_instances[0].hb_iters, 0)

        # Because n workers is now 2, we expect 2 sh intensifier_instances
        self.assertEqual(len(self.PHB.intensifier_instances), 2)

        # Each of the intensifier_instances should have s equal to 1
        # As no iteration has been completed
        self.assertEqual(self.PHB.intensifier_instances[0].s_max, 1)
        self.assertEqual(self.PHB.intensifier_instances[0].s, 1)
        self.assertEqual(self.PHB.intensifier_instances[1].s_max, 1)
        self.assertEqual(self.PHB.intensifier_instances[1].s, 1)

        # First let us check everything makes sense in HB-SH-0 HB-SH-0
        self.assertEqual(self.PHB.intensifier_instances[0].sh_intensifier.initial_budget, 2)
        self.assertEqual(self.PHB.intensifier_instances[0].sh_intensifier.max_budget, 5)
        self.assertEqual(len(self.PHB.intensifier_instances[0].sh_intensifier.n_configs_in_stage), 2)
        self.assertEqual(self.PHB.intensifier_instances[1].sh_intensifier.initial_budget, 2)
        self.assertEqual(self.PHB.intensifier_instances[1].sh_intensifier.max_budget, 5)
        self.assertEqual(len(self.PHB.intensifier_instances[1].sh_intensifier.n_configs_in_stage), 2)

        # We are running with:
        # + 4 runs for sh instance 0 ('all_budgets': array([2.5, 5. ]), 'n_configs_in_stage': [2.0, 1.0])
        #   that is, for SH0 we run in stage==0 int(2.5) intensifier_instances * 2.0 configs
        # And this times 2 because we have 2 HB intensifier_instances
        self.assertEqual(i, 8)

        # Adding a new worker is not possible as we already have 2 intensifier_instances
        # and n_workers==2
        intent, run_info = self.PHB.get_next_run(
            challengers=challengers,
            incumbent=None, chooser=None, run_history=self.rh,
            num_workers=2,
        )
        self.assertEqual(intent, RunInfoIntent.WAIT)

    def test_add_new_instance(self):
        """Test whether we can add a instance and when we should not"""

        # By default we do not create a HB
        # test adding the first instance!
        self.assertEqual(len(self.PHB.intensifier_instances), 0)
        self.assertTrue(self.PHB._add_new_instance(num_workers=1))
        self.assertEqual(len(self.PHB.intensifier_instances), 1)
        self.assertIsInstance(self.PHB.intensifier_instances[0], SuccessiveHalving)
        # A second call should not add a new HB instance
        self.assertFalse(self.PHB._add_new_instance(num_workers=1))

        # We try with 2 HB instance active

        # We effectively return true because we added a new HB instance
        self.assertTrue(self.PHB._add_new_instance(num_workers=2))

        self.assertEqual(len(self.PHB.intensifier_instances), 2)
        self.assertIsInstance(self.PHB.intensifier_instances[1], SuccessiveHalving)

        # Trying to add a third one should return false
        self.assertFalse(self.PHB._add_new_instance(num_workers=2))
        self.assertEqual(len(self.PHB.intensifier_instances), 2)

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

        # Get the run_history for a HB instance run:
        rh = RunHistory()
        stats = Stats(scenario=self.scen)
        stats.start_timing()
        HB = Hyperband(
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
        incumbent, inc_perf = self._exhaust_run_and_get_incumbent(HB, rh, num_workers=1)

        # Just to make sure nothing has changed from the HB instance side to make
        # this check invalid:
        # We add config values, so config 3 with 0 and 7 should be the lesser cost
        self.assertEqual(incumbent, self.config3)
        self.assertEqual(inc_perf, 7.0)

        # Do the same for PHB, but have multiple HB instance in there
        # This HB instance will be created via num_workers==2
        # in self._exhaust_run_and_get_incumbent
        PHB = ParallelHyperband(
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
        incumbent_phb, inc_perf_phb = self._exhaust_run_and_get_incumbent(PHB, self.rh)
        self.assertEqual(incumbent, incumbent_phb)

        # This makes sure there is a single incumbent in PHB
        self.assertEqual(inc_perf, inc_perf_phb)

        # We don't want to loose any configuration, and particularly
        # we want to make sure the values of HB instance to PHB match
        self.assertEqual(len(self.rh.data), len(rh.data))

        # Because it is a deterministic run, the run histories must be the
        # same on exhaustion
        self.assertDictEqual(self.rh.data, rh.data)


if __name__ == "__main__":
    unittest.main()
