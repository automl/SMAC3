import os
import sys
import unittest

import numpy as np

from smac.configspace import ConfigurationSpace
from smac.tae import StatusType
from smac.tae.serial_runner import SerialRunner
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.runhistory.runhistory import RunInfo

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TaeTest(unittest.TestCase):

    def setUp(self):
        self.current_dir = os.getcwd()
        base_dir = os.path.split(__file__)[0]
        base_dir = os.path.join(base_dir, '..', '..')
        os.chdir(base_dir)

    def tearDown(self):
        os.chdir(self.current_dir)

    @mock.patch.object(SerialRunner, 'run')
    def test_start_tae_return_abort(self, test_run):
        '''
            testing abort
        '''
        # Patch run-function for custom-return
        test_run.return_value = StatusType.ABORT, 12345.0, 1.2345, {}

        scen = Scenario(
            scenario={
                'cs': ConfigurationSpace(),
                'run_obj': 'quality',
                'output_dir': '',
            },
            cmd_options=None,
        )
        stats = Stats(scen)
        stats.start_timing()
        eta = SerialRunner(ta=lambda *args: None, stats=stats)

        _, run_value = eta.run_wrapper(
            RunInfo(
                config=None, instance=1, instance_specific=None,
                cutoff=30, seed=None, capped=False, budget=0.0
            )
        )
        self.assertEqual(run_value.status, StatusType.ABORT)

    @mock.patch.object(SerialRunner, 'run')
    def test_start_tae_return_nan_inf(self, test_run):
        '''
            test nan-handling and inf-handling
        '''

        def get_tae(obj):
            """ Create SerialRunner-object for testing. """
            scen = Scenario(scenario={'cs': ConfigurationSpace(), 'run_obj': obj,
                                      'cutoff_time': '10'}, cmd_options=None)
            stats = Stats(scen)
            stats.start_timing()
            # Add first run to not trigger FirstRunCrashedException
            stats.submitted_ta_runs += 1
            eta = SerialRunner(ta=lambda *args: None, stats=stats, run_obj=obj)
            return eta

        # TEST NAN
        eta = get_tae('runtime')
        # Patch run-function for custom-return (obj = runtime, cost = nan)
        test_run.return_value = StatusType.SUCCESS, np.nan, 1, {}
        run_info, result = eta.run_wrapper(RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.SUCCESS)
        #                                      (obj = runtime, runtime = nan)
        test_run.return_value = StatusType.SUCCESS, 1, np.nan, {}
        run_info, result = eta.run_wrapper(RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.CRASHED)

        eta = get_tae('quality')
        # Patch run-function for custom-return (obj = quality, cost = nan)
        test_run.return_value = StatusType.SUCCESS, np.nan, 1, {}
        run_info, result = eta.run_wrapper(RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.CRASHED)
        #                                      (obj = quality, runtime = nan)
        test_run.return_value = StatusType.SUCCESS, 1, np.nan, {}
        run_info, result = eta.run_wrapper(RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.SUCCESS)

        # TEST INF
        eta = get_tae('runtime')
        # Patch run-function for custom-return (obj = runtime, cost = inf)
        test_run.return_value = StatusType.SUCCESS, np.inf, 1, {}
        run_info, result = eta.run_wrapper(RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.SUCCESS)
        #                                      (obj = runtime, runtime = inf)
        test_run.return_value = StatusType.SUCCESS, 1, np.inf, {}
        run_info, result = eta.run_wrapper(RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.TIMEOUT)

        eta = get_tae('quality')
        # Patch run-function for custom-return (obj = quality, cost = inf)
        test_run.return_value = StatusType.SUCCESS, np.inf, 1, {}
        run_info, result = eta.run_wrapper(RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.CRASHED)
        #                                      (obj = quality, runtime = inf)
        test_run.return_value = StatusType.SUCCESS, 1, np.inf, {}
        run_info, result = eta.run_wrapper(RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.SUCCESS)

    @mock.patch.object(SerialRunner, 'run')
    def test_crashed_cost_value(self, test_run):
        '''
            test cost on crashed runs
        '''
        # Patch run-function for custom-return
        scen = Scenario(scenario={'cs': ConfigurationSpace(),
                                  'run_obj': 'quality'}, cmd_options=None)
        stats = Stats(scen)
        stats.start_timing()
        stats.submitted_ta_runs += 1

        # Check quality
        test_run.return_value = StatusType.CRASHED, np.nan, np.nan, {}
        eta = SerialRunner(ta=lambda *args: None, stats=stats,
                           run_obj='quality', cost_for_crash=100)
        run_info, result = eta.run_wrapper(RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=None, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(100, result.cost)

        # Check runtime
        eta = SerialRunner(ta=lambda *args: None, stats=stats,
                           run_obj='runtime', cost_for_crash=10.7)
        run_info, result = eta.run_wrapper(RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=20, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(20.0, result.cost)


if __name__ == "__main__":
    unittest.main()
