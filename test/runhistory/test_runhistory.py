'''
Created on Nov 19, 2015

@author: lindauer
'''
import unittest
import logging

from smac.runhistory.runhistory import RunHistory


class RunhistoryTest(unittest.TestCase):


    def test_add(self):
        '''
            simply aading some rundata to runhistory
        '''
        rh = RunHistory()
        rh.add(config={'a':'1', 'b':'2'}, cost=10, time=20,
            status="SUCCESS", instance_id=None,
            seed=None,
            additional_info=None)

        rh.add(config={'a':'1', 'b':'2'}, cost=10, time=20,
            status="SUCCESS", instance_id=1,
            seed=12354,
            additional_info={"start_time": 10})

        print(rh.data[0])
        print(rh.data[1])

if __name__ == "__main__":
    unittest.main()