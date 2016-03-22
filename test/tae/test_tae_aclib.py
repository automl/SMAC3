'''
Created on Nov 19, 2015

@author: lindauer
'''
import unittest
import shlex
import logging

from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib
from smac.tae.execute_ta_run import StatusType


class TaeOldTest(unittest.TestCase):

    def test_run(self):
        '''
            running some simple algo in aclib 2.0 style
        '''
        eta = ExecuteTARunAClib(ta=shlex.split("python dummy_ta_wrapper_aclib.py 1"))
        status, cost, runtime, ar_info = eta.run(config={})
        assert status == StatusType.TIMEOUT
        assert cost == 2.0
        assert runtime == 2.0
        
        print(status, cost, runtime)
        
        eta = ExecuteTARunAClib(ta=shlex.split("python dummy_ta_wrapper_aclib.py 2"))
        status, cost, runtime, ar_info = eta.run(config={})
        assert status == StatusType.SUCCESS
        assert cost == 3.0
        assert runtime == 3.0
        
        print(status, cost, runtime)
        
        eta = ExecuteTARunAClib(ta=shlex.split("python dummy_ta_wrapper_aclib.py 2"), run_obj="quality")
        status, cost, runtime, ar_info = eta.run(config={},)
        assert status == StatusType.SUCCESS
        assert cost == 2.0
        assert runtime == 3.0
        
        print(status, cost, runtime, ar_info)
        
        
if __name__ == "__main__":
    unittest.main()
