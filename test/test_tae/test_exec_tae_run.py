'''
Created on Jan 31, 2017

copied from test_exec_tae_old and modified to test exceptions

@author: lindauer
@modified: marben
'''
import os
import sys
import unittest

from smac.configspace import ConfigurationSpace
from smac.tae.execute_ta_run import ExecuteTARun, StatusType
from smac.tae.execute_ta_run import BudgetExhaustedException, TAEAbortException
from smac.tae.execute_ta_run import FirstRunCrashedException
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats

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
        scen = Scenario(scenario={'wallclock_limit': -1, 'cs': ConfigurationSpace()}, cmd_args=None)
        stats = Stats(scen)
        stats.start_timing()
        eta = ExecuteTARun(
                ta=lambda *args: None,  # Dummy-function
                stats=stats)

        self.assertRaises(
            BudgetExhaustedException, eta.start, config={}, instance=1)

    @mock.patch.object(ExecuteTARun, 'run')
    def test_start_tae_return_abort(self, test_run):
        '''
            testing abort
        '''
        # Patch run-function for custom-return
        test_run.return_value = StatusType.ABORT, 12345.0, 1.2345, {}

        scen = Scenario(scenario={'cs': ConfigurationSpace()}, cmd_args=None)
        stats = Stats(scen)
        stats.start_timing()
        eta = ExecuteTARun(ta=lambda *args: None, stats=stats)

        self.assertRaises(
            TAEAbortException, eta.start, config={}, instance=1)

    @mock.patch.object(ExecuteTARun, 'run')
    def test_start_crash_first_run(self, test_run):
        '''
            testing crash-on-first-run
        '''
        # Patch run-function for custom-return
        test_run.return_value = StatusType.CRASHED, 12345.0, 1.2345, {}

        scen = Scenario(scenario={'cs': ConfigurationSpace()}, cmd_args=None)
        stats = Stats(scen)
        stats.start_timing()
        eta = ExecuteTARun(ta=lambda *args: None, stats=stats)

        self.assertRaises(
            FirstRunCrashedException, eta.start, config={}, instance=1)


if __name__ == "__main__":
    unittest.main()
