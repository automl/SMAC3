'''
Created on Mar 29, 2015

@author: Andre Biedenkapp
'''
import unittest
import logging
import numpy as np

from smac.scenario.scenario import Scenario

class ScenarioTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger('ScenarioTest')
        self.logger.setLevel(logging.DEBUG)
        
        self.test_scenario_dict = {'algo': 'echo Hello',
                                   'paramfile': 'test/test_files/scenario_test/param.pcs',
                                   'execdir': '.',
                                   'deterministic': 0,
                                   'run_obj': 'runtime',
                                   'overall_obj': 'mean10',
                                   'cutoff_time': 5,
                                   'wallclock-limit': 18000,
                                   'instance_file': 'test/test_files/scenario_test/training.txt',
                                   'test_instance_file': 'test/test_files/scenario_test/test.txt',
                                   'feature_file': 'test/test_files/scenario_test/features.txt'}

    def test_Exception(self):
        with self.assertRaises(TypeError):
            s = Scenario(['a', 'b'])
    
    def test_string_scenario(self):
        scenario = Scenario('test/test_files/scenario_test/scenario.txt')
        
        self.assertEquals(scenario.ta, ['echo', 'Hello'])
        self.assertEquals(scenario.execdir, '.')
        self.assertFalse(scenario.deterministic)
        self.assertEquals(scenario.pcs_fn, 'test/test_files/scenario_test/param.pcs')
        self.assertEquals(scenario.overall_obj, 'mean10')
        self.assertEquals(scenario.cutoff, 5.)
        self.assertEquals(scenario.algo_runs_timelimit, np.inf)
        self.assertEquals(scenario.wallclock_limit, 18000)
        self.assertEquals(scenario.par_factor, 10)
        self.assertEquals(scenario.train_insts, ['d', 'e', 'f'])
        self.assertEquals(scenario.test_insts, ['a', 'b', 'c'])
        test_dict = {'d' : 1, 'e' : 2, 'f' : 3}
        self.assertEquals(scenario.feature_dict, test_dict)
        self.assertEquals(scenario.feature_array[0], 1)

    def test_dict_scenario(self):
        scenario = Scenario(self.test_scenario_dict)
        
        self.assertEquals(scenario.ta, ['echo', 'Hello'])
        self.assertEquals(scenario.execdir, '.')
        self.assertFalse(scenario.deterministic)
        self.assertEquals(scenario.pcs_fn, 'test/test_files/scenario_test/param.pcs')
        self.assertEquals(scenario.overall_obj, 'mean10')
        self.assertEquals(scenario.cutoff, 5.)
        self.assertEquals(scenario.algo_runs_timelimit, np.inf)
        self.assertEquals(scenario.wallclock_limit, 18000)
        self.assertEquals(scenario.par_factor, 10)
        self.assertEquals(scenario.train_insts, ['d', 'e', 'f'])
        self.assertEquals(scenario.test_insts, ['a', 'b', 'c'])
        test_dict = {'d' : 1, 'e' : 2, 'f' : 3}
        self.assertEquals(scenario.feature_dict, test_dict)
        self.assertEquals(scenario.feature_array[0], 1)

if __name__ == "__main__":
    unittest.main()
