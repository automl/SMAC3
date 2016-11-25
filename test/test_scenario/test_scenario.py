'''
Created on Mar 29, 2015

@author: Andre Biedenkapp
'''
from collections import defaultdict
import os
import logging
import unittest
import pickle
import copy

import numpy as np

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.scenario.scenario import Scenario
from smac.configspace import ConfigurationSpace
from smac.utils.merge_foreign_data import merge_foreign_data
from smac.runhistory.runhistory import RunHistory
from smac.smbo.objective import average_cost
from smac.tae.execute_ta_run import StatusType


class InitFreeScenario(Scenario):

    def __init__(self):
        self._groups = defaultdict(set)


class ScenarioTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger('ScenarioTest')
        self.logger.setLevel(logging.DEBUG)

        base_directory = os.path.split(__file__)[0]
        base_directory = os.path.abspath(
            os.path.join(base_directory, '..', '..'))
        self.current_dir = os.getcwd()
        os.chdir(base_directory)

        self.cs = ConfigurationSpace()

        self.test_scenario_dict = {'algo': 'echo Hello',
                                   'paramfile':
                                       'test/test_files/scenario_test/param.pcs',
                                   'execdir': '.',
                                   'deterministic': 0,
                                   'run_obj': 'runtime',
                                   'overall_obj': 'mean10',
                                   'cutoff_time': 5,
                                   'wallclock-limit': 18000,
                                   'instance_file':
                                       'test/test_files/scenario_test/training.txt',
                                   'test_instance_file':
                                       'test/test_files/scenario_test/test.txt',
                                   'feature_file':
                                       'test/test_files/scenario_test/features.txt'}

    def tearDown(self):
        os.chdir(self.current_dir)

    def test_Exception(self):
        with self.assertRaises(TypeError):
            s = Scenario(['a', 'b'])

    def test_string_scenario(self):
        scenario = Scenario('test/test_files/scenario_test/scenario.txt')

        self.assertEquals(scenario.ta, ['echo', 'Hello'])
        self.assertEquals(scenario.execdir, '.')
        self.assertFalse(scenario.deterministic)
        self.assertEquals(
            scenario.pcs_fn, 'test/test_files/scenario_test/param.pcs')
        self.assertEquals(scenario.overall_obj, 'mean10')
        self.assertEquals(scenario.cutoff, 5.)
        self.assertEquals(scenario.algo_runs_timelimit, np.inf)
        self.assertEquals(scenario.wallclock_limit, 18000)
        self.assertEquals(scenario.par_factor, 10)
        self.assertEquals(scenario.train_insts, ['d', 'e', 'f'])
        self.assertEquals(scenario.test_insts, ['a', 'b', 'c'])
        test_dict = {'d': 1, 'e': 2, 'f': 3}
        self.assertEquals(scenario.feature_dict, test_dict)
        self.assertEquals(scenario.feature_array[0], 1)

    def test_dict_scenario(self):
        scenario = Scenario(self.test_scenario_dict)

        self.assertEquals(scenario.ta, ['echo', 'Hello'])
        self.assertEquals(scenario.execdir, '.')
        self.assertFalse(scenario.deterministic)
        self.assertEquals(
            scenario.pcs_fn, 'test/test_files/scenario_test/param.pcs')
        self.assertEquals(scenario.overall_obj, 'mean10')
        self.assertEquals(scenario.cutoff, 5.)
        self.assertEquals(scenario.algo_runs_timelimit, np.inf)
        self.assertEquals(scenario.wallclock_limit, 18000)
        self.assertEquals(scenario.par_factor, 10)
        self.assertEquals(scenario.train_insts, ['d', 'e', 'f'])
        self.assertEquals(scenario.test_insts, ['a', 'b', 'c'])
        test_dict = {'d': 1, 'e': 2, 'f': 3}
        self.assertEquals(scenario.feature_dict, test_dict)
        self.assertEquals(scenario.feature_array[0], 1)

    def unknown_parameter_in_scenario(self):
        self.assertRaisesRegex(ValueError,
                               'Could not parse the following arguments: '
                               'duairznbvulncbzpneairzbnuqdae',
                               Scenario,
                               {'wallclock-limit': '12345',
                                'duairznbvulncbzpneairzbnuqdae': 'uqpab'})

    def test_add_argument(self):
        # Check if adding non-bool required fails
        scenario = InitFreeScenario()
        self.assertRaisesRegexp(TypeError, "Argument 'required' must be of "
                                           "type 'bool'.",
                                scenario.add_argument, 'name', required=1,
                                help=None)

        # Check that adding a parameter which is both required and in a
        # required-group fails
        self.assertRaisesRegexp(ValueError, "Cannot make argument 'name' "
                                            "required and add it to a group of "
                                            "mutually exclusive arguments.",
                                scenario.add_argument, 'name', required=True,
                                help=None, mutually_exclusive_group='required')

    def test_merge_foreign_data(self):
        ''' test smac.utils.merge_foreign_data '''

        scenario = Scenario(self.test_scenario_dict)
        scenario_2 = Scenario(self.test_scenario_dict)
        scenario_2.feature_dict = {"inst_new": [4]}

        # init cs
        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformIntegerHyperparameter(name='a',
                                                           lower=0,
                                                           upper=100))
        cs.add_hyperparameter(UniformIntegerHyperparameter(name='b',
                                                           lower=0,
                                                           upper=100))
        # build runhistory
        rh_merge = RunHistory(aggregate_func=average_cost)
        config = Configuration(cs, values={'a': 1, 'b': 2})

        rh_merge.add(config=config, instance_id="inst_new", cost=10, time=20,
                     status=StatusType.SUCCESS,
                     seed=None,
                     additional_info=None)
        
        # "d" is an instance in <scenario>
        rh_merge.add(config=config, instance_id="d", cost=5, time=20,
                     status=StatusType.SUCCESS,
                     seed=None,
                     additional_info=None)

        # build empty rh
        rh_base = RunHistory(aggregate_func=average_cost)

        merge_foreign_data(scenario=scenario, runhistory=rh_base,
                           in_scenario_list=[scenario_2], in_runhistory_list=[rh_merge])

        # both runs should be in the runhistory
        # but only the run on "d" should be used for the cost of <config>
        self.assertTrue(len(rh_base.data) == 2)
        self.assertTrue(rh_base.get_cost(config) == 5)
        

        rh_merge.add(config=config, instance_id="inst_new_2", cost=10, time=20,
                     status=StatusType.SUCCESS,
                     seed=None,
                     additional_info=None)
        
        self.assertRaises(ValueError, merge_foreign_data, **{"scenario":scenario, "runhistory":rh_base, "in_scenario_list":[scenario_2], "in_runhistory_list":[rh_merge]})

    def test_pickle_dump(self):
        scenario = Scenario(self.test_scenario_dict)

        packed_scenario = pickle.dumps(scenario)
        self.assertIsNotNone(packed_scenario)

        unpacked_scenario = pickle.loads(packed_scenario)
        self.assertIsNotNone(unpacked_scenario)
        self.assertIsNotNone(unpacked_scenario.logger)
        self.assertEqual(scenario.logger.name, unpacked_scenario.logger.name)

    def test_choice_argument(self):
        scenario_dict = self.test_scenario_dict
        scenario_dict['initial_incumbent'] = 'DEFAULT'
        scenario = Scenario(scenario_dict)
        self.assertEqual(scenario.initial_incumbent, 'DEFAULT')

        self.assertRaisesRegex(TypeError, 'Choice must be of type '
                                          'list/set/tuple.',
                               scenario.add_argument, name='a', choice='abc',
                               help=None)

        scenario_dict['initial_incumbent'] = 'Default'
        self.assertRaisesRegex(ValueError,
                               "Argument initial_incumbent can only take a "
                               "value in ['DEFAULT, 'RANDOM'] but is Default")


if __name__ == "__main__":
    unittest.main()
