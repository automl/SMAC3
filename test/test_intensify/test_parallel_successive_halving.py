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
from smac.intensification.parallel_successive_halving import ParallelSuccessiveHalving
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


class TestSuccessiveHalving(unittest.TestCase):

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

        self.scen = Scenario({"cutoff_time": 2, 'cs': self.cs,
                              "run_obj": 'runtime',
                              "output_dir": ''})
        self.stats = Stats(scenario=self.scen)
        self.stats.start_timing()

        # Create the base object
        self.PSH = ParallelSuccessiveHalving(
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
            max_active_SH=2,
        )

    def test_initialization(self):
        """Makes sure that a proper SH is created"""

        # One SH instance gets created at least
        self.assertEqual(len(self.PSH.sh_instances), 1)

        # Parameters properly passed to SH
        self.assertEqual(len(self.PSH.sh_instances[0].inst_seed_pairs), 10)
        self.assertEqual(self.PSH.sh_instances[0].initial_budget, 2)
        self.assertEqual(self.PSH.sh_instances[0].max_budget, 5)

    def test_process_results_via_sourceid(self):
        """Makes sure source id is honored when deciding
        which SH will consume the result/run_info"""
        # Mock the SH so we can make sure the correct item is passed
        for i in range(10):
            self.PSH.sh_instances[i] = mock.Mock()

        # randomly create run_infos and push into PSH. Then we will make
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
            # That is we check only the proper SH has this
            magic = time.time()

            result = RunValue(
                cost=1,
                time=0.5,
                status=StatusType.SUCCESS,
                starttime=1,
                endtime=2,
                additional_info=magic
            )
            self.PSH.process_results(
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
                self.PSH.sh_instances[i].process_results.call_args[1]['run_info'], run_info)
            self.assertEqual(
                self.PSH.sh_instances[i].process_results.call_args[1]['result'], result)
            all_other_run_infos, all_other_results = [], []
            for j in range(len(self.PSH.sh_instances)):
                # Skip the expected SH
                if i == j:
                    continue
                if self.PSH.sh_instances[j].process_results.call_args is None:
                    all_other_run_infos.append(None)
                else:
                    all_other_run_infos.append(
                        self.PSH.sh_instances[j].process_results.call_args[1]['run_info'])
                    all_other_results.append(
                        self.PSH.sh_instances[j].process_results.call_args[1]['result'])
            self.assertNotIn(run_info, all_other_run_infos)
            self.assertNotIn(result, all_other_results)

    def test_get_next_run_single_SH(self):
        """Makes sure that a single SH returns a valid config"""

        # Everything here will be tested with a single SH
        self.PSH.max_active_SH = 1

        challengers = [self.config1, self.config2, self.config3, self.config4]
        for i in range(30):
            intent, run_info = self.PSH.get_next_run(
                challengers=challengers,
                incumbent=None, chooser=None, run_history=self.rh)

            # Regenerate challenger list
            challengers = [c for c in challengers if c != run_info.config]

            if intent == RunInfoIntent.WAIT:
                break

            # Add the config to self.rh in order to make PSH aware that this
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

        # We should not create more SH instances
        self.assertEqual(len(self.PSH.sh_instances), 1)

        # We are running with:
        # 'all_budgets': array([2.5, 5. ]) -> 2 instances per config top
        # 'n_configs_in_stage': [2.0, 1.0],
        # This means we run int(2.5) + 2.0 = 4 runs before waiting
        self.assertEqual(i, 4)

    def test_get_next_run_dual_SH(self):
        """Makes sure that two  SH can properly coexist and tag
        run_info properly"""

        # Everything here will be tested with a single SH
        self.PSH.max_active_SH = 2

        challengers = [self.config1, self.config2, self.config3, self.config4]
        for i in range(30):
            intent, run_info = self.PSH.get_next_run(
                challengers=challengers,
                incumbent=None, chooser=None, run_history=self.rh)

            # Regenerate challenger list
            challengers = [c for c in challengers if c != run_info.config]

            # Add the config to self.rh in order to make PSH aware that this
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

        # We create a second sh instances as after 4 runs, the SH
        # number zero needs to wait
        self.assertEqual(len(self.PSH.sh_instances), 2)

        # We are running with:
        # 'all_budgets': array([2.5, 5. ]) -> 2 instances per config top
        # 'n_configs_in_stage': [2.0, 1.0],
        # This means we run int(2.5) + 2.0 = 4 runs before waiting
        # But we have 2 successive halvers now!
        self.assertEqual(i, 8)

    def test_add_new_SH(self):
        """Test whether we can add a SH and when we should not"""

        # Case of a single SH
        self.PSH.max_active_SH = 1

        # By default we create at least 1 SH
        self.assertEqual(len(self.PSH.sh_instances), 1)
        self.assertIsInstance(self.PSH.sh_instances[0], SuccessiveHalving)

        # If max active SH is called, we should not add another SH
        # and return false
        self.assertFalse(self.PSH._add_new_SH())
        self.assertEqual(len(self.PSH.sh_instances), 1)

        # We try with 2 SH active
        self.PSH.max_active_SH = 2

        # We effectively return true because we added a new SH
        self.assertTrue(self.PSH._add_new_SH())

        self.assertEqual(len(self.PSH.sh_instances), 2)
        self.assertIsInstance(self.PSH.sh_instances[1], SuccessiveHalving)

        # Trying to add a third one should return false
        self.assertFalse(self.PSH._add_new_SH())
        self.assertEqual(len(self.PSH.sh_instances), 2)

    def test_sh_scheduling(self):
        """Ensures proper scheduling is returned based on the strategy"""

        # Add more SH to make testing interesting
        self.PSH.max_active_SH = 4
        for i in range(4):
            self.PSH._add_new_SH()

        # First test the ordinal approach
        self.assertCountEqual(list(self.PSH._sh_scheduling(strategy='ordinal')), [0, 1, 2, 3])

        # Also, if strategy is serial and index is 3 it should be the same as ordinal
        # That is, we lastly queried the SH #3, we have to start from 0 again
        self.PSH.last_active_instance = 3
        self.assertCountEqual(list(self.PSH._sh_scheduling(strategy='serial')), [0, 1, 2, 3])

        # Make sure we complete the list properly in serial
        self.PSH.last_active_instance = 0
        self.assertCountEqual(list(self.PSH._sh_scheduling(strategy='serial')), [1, 2, 3, 0])
        self.PSH.last_active_instance = 1
        self.assertCountEqual(list(self.PSH._sh_scheduling(strategy='serial')), [2, 3, 0, 1])
        self.PSH.last_active_instance = 2
        self.assertCountEqual(list(self.PSH._sh_scheduling(strategy='serial')), [3, 0, 1, 2])

    def _exhaust_run_and_get_incumbent(self, sh, rh):
        """
        Runs all provided configs on all instances and return the incumbent
        as a nice side effect runhistory/stats are properly filled
        """
        challengers = [self.config1, self.config2, self.config3, self.config4]
        incumbent = None
        for i in range(100):
            try:
                intent, run_info = sh.get_next_run(
                    challengers=challengers,
                    incumbent=None, chooser=None, run_history=rh)
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

    def test_parallel_same_as_serial_SH(self):
        """Makes sure we behave the same as a serial run at the end"""

        # Get the run_history for a SH run:
        rh = RunHistory()
        stats = Stats(scenario=self.scen)
        stats.start_timing()
        SH = SuccessiveHalving(
            stats=stats,
            traj_logger=TrajLogger(output_dir=None, stats=stats),
            rng=np.random.RandomState(12345),
            deterministic=False,
            run_obj_time=False,
            instances=[1, 2, 3, 4, 5],
            n_seeds=2,
            initial_budget=2,
            max_budget=5,
            eta=2,
        )
        incumbent, inc_perf = self._exhaust_run_and_get_incumbent(SH, rh)

        # Just to make sure nothing has changed form SH side to make
        # this check invalid:
        # We add config values, so config 4 with 0 and 0 should be the lesser cost
        self.assertEqual(incumbent, self.config3)
        self.assertEqual(inc_perf, 7.0)

        # Do the same for PSH, but have multiple SH in there
        # _add_new_SH returns true if it was able to add a new SH
        # scheduling is defaulted to serial
        self.assertTrue(self.PSH._add_new_SH())
        incumbent_psh, inc_perf_psh = self._exhaust_run_and_get_incumbent(self.PSH, self.rh)
        self.assertEqual(incumbent, incumbent_psh)

        # This makes sure there is a single incumbent in PSH
        self.assertEqual(inc_perf, inc_perf_psh)

        # We don't want to loose any configuration, and particularly
        # we want to make sure the values of SH to PSH match
        self.assertEqual(len(self.rh.data), len(rh.data))

        configs_sh_rh = {}
        for k, v in rh.data.items():
            config_sh = rh.ids_config[k.config_id]
            if config_sh not in configs_sh_rh:
                configs_sh_rh[config_sh] = []
            configs_sh_rh[config_sh].append(v.cost)
        configs_psh_rh = {}
        for k, v in rh.data.items():
            config_sh = rh.ids_config[k.config_id]
            if config_sh not in configs_psh_rh:
                configs_psh_rh[config_sh] = []
            configs_psh_rh[config_sh].append(v.cost)

        # If this dictionaries are equal it means we have all configs
        # and the values track the numbers and actual cost!
        self.assertDictEqual(configs_sh_rh, configs_psh_rh)


if __name__ == "__main__":
    unittest.main()
