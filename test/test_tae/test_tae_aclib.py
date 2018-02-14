'''
Created on Nov 19, 2015

@author: lindauer
'''
import os
import unittest
import shlex

from smac.configspace import ConfigurationSpace
from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib
from smac.tae.execute_ta_run import StatusType
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats


class TaeOldTest(unittest.TestCase):

    def setUp(self):
        self.current_dir = os.getcwd()
        base_dir = os.path.split(__file__)[0]
        base_dir = os.path.join(base_dir, '..', '..')
        os.chdir(base_dir)

    def tearDown(self):
        os.chdir(self.current_dir)

    def test_run(self):
        '''
            running some simple algo in aclib 2.0 style
        '''
        scen = Scenario(scenario={'cs': ConfigurationSpace(),
                                  'run_obj': 'quality',
                                  'output_dir': ''}, cmd_options=None)
        stats = Stats(scen)

        eta = ExecuteTARunAClib(
            ta=shlex.split("python test/test_tae/dummy_ta_wrapper_aclib.py 1"),
            stats=stats)
        status, cost, runtime, ar_info = eta.run(config={})
        assert status == StatusType.TIMEOUT
        assert cost == 2.0
        assert runtime == 2.0

        print(status, cost, runtime)

        eta = ExecuteTARunAClib(
            ta=shlex.split("python test/test_tae/dummy_ta_wrapper_aclib.py 2"),
            stats=stats)
        status, cost, runtime, ar_info = eta.run(config={})
        assert status == StatusType.SUCCESS
        assert cost == 3.0
        assert runtime == 3.0

        print(status, cost, runtime)

        eta = ExecuteTARunAClib(ta=shlex.split(
            "python test/test_tae/dummy_ta_wrapper_aclib.py 2"), stats=stats,
            run_obj="quality")
        status, cost, runtime, ar_info = eta.run(config={},)
        assert status == StatusType.SUCCESS
        assert cost == 2.0
        assert runtime == 3.0

        print(status, cost, runtime, ar_info)
        
    def test_status(self):
        
        scen = Scenario(scenario={'cs': ConfigurationSpace(),
                                  'run_obj': 'quality',
                                  'output_dir': ''}, cmd_options=None)
        stats = Stats(scen)

        eta = ExecuteTARunAClib(
            ta=shlex.split(""),
            stats=stats)
        
        def test_success(**kwargs):
            results = {"status": "SUCCESS",
                       "cost": 1234567890
                       }
            return results, "", ""
        
        eta._call_ta = test_success
        status, cost, runtime, ar_info = eta.run(config={},)
        self.assertEqual(status, StatusType.SUCCESS)
        
        def test_success(**kwargs):
            results = {"status": "SUCESS",
                       "cost": 1234567890
                       }
            return results, "", ""
        
        eta._call_ta = test_success
        status, cost, runtime, ar_info = eta.run(config={},)
        self.assertEqual(status, StatusType.CRASHED)
        
        def test_success(**kwargs):
            results = {"status": "success",
                       "cost": 1234567890
                       }
            return results, "", ""
        
        eta._call_ta = test_success
        status, cost, runtime, ar_info = eta.run(config={},)
        self.assertEqual(status, StatusType.CRASHED)


if __name__ == "__main__":
    unittest.main()
