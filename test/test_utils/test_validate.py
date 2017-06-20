import unittest
import logging

import numpy as np

try:
    import unittest.mock
    from unittest.mock import patch
except:
    from mock import patch

from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import get_runs
from smac.utils.validate import validate

class ValidationTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.value = 0
        self.rng = np.random.RandomState(seed=42)
        self.scenario = Scenario({'run_obj': 'quality', 'param_file':
                                  'test/test_files/validation/test_validation_pcs.pcs',
                                  'algo': 'python -u test/test_files/example_ta.py'
                                 })
        self.scenario.train_insts = {'0': 'null', '1': 'one', '2': 'two'}
        self.scenario.test_insts = {'3': 'three', '4': 'four', '5': 'five'}
        self.trajectory = TrajLogger.read_traj_aclib_format(
            fn='test/test_files/validation/test_validation_traj.json', cs=self.scenario.cs)

    def test_get_runs(self):
        ''' test if the runs are generated as expected '''
        # Get multiple configs
        expected = [{'inst_specs': 'three', 'seed': 1608637542, 'inst': '3', 'config': 'config1'},
                    {'inst_specs': 'three', 'seed': 1608637542, 'inst': '3', 'config': 'config2'},
                    {'inst_specs': 'three', 'seed': 1273642419, 'inst': '3', 'config': 'config1'},
                    {'inst_specs': 'three', 'seed': 1273642419, 'inst': '3', 'config': 'config2'},
                    {'inst_specs': 'four',  'seed': 1935803228, 'inst': '4', 'config': 'config1'},
                    {'inst_specs': 'four',  'seed': 1935803228, 'inst': '4', 'config': 'config2'},
                    {'inst_specs': 'four',  'seed': 787846414,  'inst': '4', 'config': 'config1'},
                    {'inst_specs': 'four',  'seed': 787846414,  'inst': '4', 'config': 'config2'},
                    {'inst_specs': 'five',  'seed': 996406378,  'inst': '5', 'config': 'config1'},
                    {'inst_specs': 'five',  'seed': 996406378,  'inst': '5', 'config': 'config2'},
                    {'inst_specs': 'five',  'seed': 1201263687, 'inst': '5', 'config': 'config1'},
                    {'inst_specs': 'five',  'seed': 1201263687, 'inst': '5', 'config': 'config2'}]

        runs = get_runs(['config1', 'config2'], self.scenario,
                        self.rng, train=False, test=True, repetitions=2)
        self.assertEqual(runs, expected)

        # Only train
        expected = [{'inst_specs': 'null', 'seed': 423734972,  'inst': '0', 'config': 'config1'},
                    {'inst_specs': 'null', 'seed': 415968276,  'inst': '0', 'config': 'config1'},
                    {'inst_specs': 'one',  'seed': 670094950,  'inst': '1', 'config': 'config1'},
                    {'inst_specs': 'one',  'seed': 1914837113, 'inst': '1', 'config': 'config1'},
                    {'inst_specs': 'two',  'seed': 669991378,  'inst': '2', 'config': 'config1'},
                    {'inst_specs': 'two',  'seed': 429389014,  'inst': '2', 'config': 'config1'}]

        runs = get_runs(['config1'], self.scenario,
                        self.rng, train=True, test=False, repetitions=2)
        self.assertEqual(runs, expected)

        # Test and train
        expected = [{'inst': '0', 'seed': 249467210,  'config': 'config1', 'inst_specs': 'null'},
                    {'inst': '1', 'seed': 1972458954, 'config': 'config1', 'inst_specs': 'one'},
                    {'inst': '2', 'seed': 1572714583, 'config': 'config1', 'inst_specs': 'two'},
                    {'inst': '3', 'seed': 1433267572, 'config': 'config1', 'inst_specs': 'three'},
                    {'inst': '4', 'seed': 434285667,  'config': 'config1', 'inst_specs': 'four'},
                    {'inst': '5', 'seed': 613608295,  'config': 'config1', 'inst_specs': 'five'}]
        runs = get_runs(['config1'], self.scenario,
                        self.rng, train=True, test=True, repetitions=1)
        self.assertEqual(runs, expected)

    def test_validate(self):
        ''' test validation '''
        # Test basic usage
        validate(self.scenario, self.trajectory, self.rng,
                 output='test/test_files/validation/test_validate_rh.json',
                 config_mode='def', instances='test',
                 repetitions=3)
        validate(self.scenario, self.trajectory, self.rng,
                 output='test/test_files/validation/test_validate_rh.json',
                 config_mode='inc', instances='train+test')
        validate(self.scenario, self.trajectory, self.rng,
                 output='test/test_files/validation/test_validate_rh.json',
                 config_mode='time', instances='train')

    def test_parallel(self):
        ''' test parallel'''
        validate(self.scenario, self.trajectory, self.rng,
                 output='test/test_files/validation/test_validate_rh.json',
                 config_mode='time', instances='train+test', n_jobs=-1)

    def test_passed_runhistory(self):
        ''' test if passed runhistory is in resulting runhistory '''
        #TODO
        pass

    def test_passed_tae(self):
        ''' test if passed tae is working '''
        #TODO
        pass
