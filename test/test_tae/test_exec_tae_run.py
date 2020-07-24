'''
Created on Jan 31, 2017

copied from test_exec_tae_old and modified to test exceptions

@author: lindauer
@modified: marben
'''
import os
import sys
import unittest

import numpy as np

from smac.configspace import ConfigurationSpace
from smac.tae.execute_ta_run import ExecuteTARun, StatusType
from smac.tae.execute_ta_run import TAEAbortException
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run_wrapper import execute_ta_run_wrapper
from smac.runhistory.runhistory import RunInfo

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock


class TaeTest(unittest.TestCase):

    def setUp(self):
        self.current_dir = os.getcwd()
        base_dir = os.path.split(__file__)[0]
        base_dir = os.path.join(base_dir, '..', '..')
        os.chdir(base_dir)

    def tearDown(self):
        os.chdir(self.current_dir)

    def test_start_exhausted_budget(self):
        '''
            testing exhausted budget
        '''
        # Set time-limit negative in scenario-options to trigger exception
        scen = Scenario(scenario={'wallclock_limit': -1, 'cs': ConfigurationSpace(),
                                  'run_obj': 'quality',
                                  'output_dir': ''}, cmd_options=None)
        stats = Stats(scen)
        stats.start_timing()
        eta = ExecuteTARun(ta=lambda *args: None,  # Dummy-function
                           stats=stats)

        # Dummy run. When on budget exhausted, the smbo
        # loop is notified via the result status
        result = execute_ta_run_wrapper(eta, RunInfo(
            config=None, instance=None, instance_specific=None,
            cutoff=None, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.BUDGETEXHAUSTED)

    @mock.patch.object(ExecuteTARun, 'run')
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
        eta = ExecuteTARun(ta=lambda *args: None, stats=stats)

        with self.assertRaises(TAEAbortException):
            execute_ta_run_wrapper(
                eta, RunInfo(
                    config=None, instance=1, instance_specific=None,
                    cutoff=30, seed=None, capped=False, budget=0.0
                )
            )

    @mock.patch.object(ExecuteTARun, 'run')
    def test_start_tae_return_nan_inf(self, test_run):
        '''
            test nan-handling and inf-handling
        '''

        def get_tae(obj):
            """ Create ExecuteTARun-object for testing. """
            scen = Scenario(scenario={'cs': ConfigurationSpace(), 'run_obj': obj,
                                      'cutoff_time': '10'}, cmd_options=None)
            stats = Stats(scen)
            stats.start_timing()
            # Add first run to not trigger FirstRunCrashedException
            stats.ta_runs += 1
            eta = ExecuteTARun(ta=lambda *args: None, stats=stats, run_obj=obj)
            return eta

        # TEST NAN
        eta = get_tae('runtime')
        # Patch run-function for custom-return (obj = runtime, cost = nan)
        test_run.return_value = StatusType.SUCCESS, np.nan, 1, {}
        result = execute_ta_run_wrapper(eta, RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.SUCCESS)
        #                                      (obj = runtime, runtime = nan)
        test_run.return_value = StatusType.SUCCESS, 1, np.nan, {}
        result = execute_ta_run_wrapper(eta, RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.CRASHED)

        eta = get_tae('quality')
        # Patch run-function for custom-return (obj = quality, cost = nan)
        test_run.return_value = StatusType.SUCCESS, np.nan, 1, {}
        result = execute_ta_run_wrapper(eta, RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.CRASHED)
        #                                      (obj = quality, runtime = nan)
        test_run.return_value = StatusType.SUCCESS, 1, np.nan, {}
        result = execute_ta_run_wrapper(eta, RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.SUCCESS)

        # TEST INF
        eta = get_tae('runtime')
        # Patch run-function for custom-return (obj = runtime, cost = inf)
        test_run.return_value = StatusType.SUCCESS, np.inf, 1, {}
        result = execute_ta_run_wrapper(eta, RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.SUCCESS)
        #                                      (obj = runtime, runtime = inf)
        test_run.return_value = StatusType.SUCCESS, 1, np.inf, {}
        result = execute_ta_run_wrapper(eta, RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.TIMEOUT)

        eta = get_tae('quality')
        # Patch run-function for custom-return (obj = quality, cost = inf)
        test_run.return_value = StatusType.SUCCESS, np.inf, 1, {}
        result = execute_ta_run_wrapper(eta, RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.CRASHED)
        #                                      (obj = quality, runtime = inf)
        test_run.return_value = StatusType.SUCCESS, 1, np.inf, {}
        result = execute_ta_run_wrapper(eta, RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=10, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(result.status, StatusType.SUCCESS)

    @mock.patch.object(ExecuteTARun, 'run')
    def test_crashed_cost_value(self, test_run):
        '''
            test cost on crashed runs
        '''
        # Patch run-function for custom-return
        scen = Scenario(scenario={'cs': ConfigurationSpace(),
                                  'run_obj': 'quality'}, cmd_options=None)
        stats = Stats(scen)
        stats.start_timing()
        stats.ta_runs += 1

        # Check quality
        test_run.return_value = StatusType.CRASHED, np.nan, np.nan, {}
        eta = ExecuteTARun(ta=lambda *args: None, stats=stats,
                           run_obj='quality', cost_for_crash=100)
        result = execute_ta_run_wrapper(eta, RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=None, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(100, result.cost)

        # Check runtime
        eta = ExecuteTARun(ta=lambda *args: None, stats=stats,
                           run_obj='runtime', cost_for_crash=10.7)
        result = execute_ta_run_wrapper(eta, RunInfo(
            config={}, instance=1, instance_specific="0",
            cutoff=20, seed=None, capped=False, budget=0.0
        ))
        self.assertEqual(20.0, result.cost)


if __name__ == "__main__":
    unittest.main()
