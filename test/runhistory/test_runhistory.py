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
        rh.add(config={'a': '1', 'b': '2'}, cost=10, time=20,
               status="SUCCESS", instance_id=None,
               seed=None,
               additional_info=None)

        rh.add(config={'a': '1', 'b': '2'}, cost=10, time=20,
               status="SUCCESS", instance_id=1,
               seed=12354,
               additional_info={"start_time": 10})

        print(rh.data)

    def test_get_config_runs(self):
        '''
            get some config runs from runhistory
        '''

        rh = RunHistory()

        rh.add(config={'a': '1', 'b': '2'}, cost=10, time=20,
               status="SUCCESS", instance_id=1,
               seed=1)

        rh.add(config={'a': '1', 'b': '3'}, cost=10, time=20,
               status="SUCCESS", instance_id=1,
               seed=1)

        rh.add(config={'a': '1', 'b': '2'}, cost=10, time=20,
               status="SUCCESS", instance_id=2,
               seed=2)

        ist = rh.get_runs_for_config(config={'a': '1', 'b': '2'})
        #print(ist)
        #print(ist[0])
        #print(ist[1])
        assert len(ist) == 2

if __name__ == "__main__":
    unittest.main()
