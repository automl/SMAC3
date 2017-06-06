'''
Created on Mar 29, 2015

@author: Andre Biedenkapp
'''
from collections import defaultdict
import sys
import os
import logging
import unittest
import pickle
import copy
import shutil

import numpy as np

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.scenario.scenario import Scenario, _is_truthy
from smac.configspace import ConfigurationSpace
from smac.utils.merge_foreign_data import merge_foreign_data
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.tae.execute_ta_run import StatusType

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock


class InitFreeScenario(Scenario):

    def __init__(self):
        self._groups = defaultdict(set)


class ScenarioTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger(
            self.__module__ + '.' + self.__class__.__name__)
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
                                       'test/test_files/scenario_test/features.txt',
                                   'output_dir':
                                       'test/test_files/scenario_test/tmp_output'}

    def tearDown(self):
        os.chdir(self.current_dir)
        shutil.rmtree(self.test_scenario_dict['output_dir'], ignore_errors=True)

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
        # but we should not use the data to update the cost of config
        self.assertTrue(len(rh_base.data) == 2)
        self.assertTrue(np.isnan(rh_base.get_cost(config)))

        # we should not get direct access to external run data
        runs = rh_base.get_runs_for_config(config)
        self.assertTrue(len(runs) == 0)

        rh_merge.add(config=config, instance_id="inst_new_2", cost=10, time=20,
                     status=StatusType.SUCCESS,
                     seed=None,
                     additional_info=None)

        self.assertRaises(ValueError, merge_foreign_data, **{
                          "scenario": scenario, "runhistory": rh_base, "in_scenario_list": [scenario_2], "in_runhistory_list": [rh_merge]})

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

    def test_write(self):
        """ Test whether a reloaded scenario still holds all the necessary
        information. A subset of parameters might change, such as the paths to
        pcs- or instance-files, so they are checked manually. """

        def check_scen_eq(scen1, scen2):
            """ Customized check for scenario-equality, ignoring file-paths """
            for name in scen1._arguments:
                dest = scen1._arguments[name]['dest']
                name = dest if dest else name  # if 'dest' is None, use 'name'
                if name in ["pcs_fn", "train_inst_fn", "test_inst_fn",
                            "feature_fn", "output_dir"]:
                    continue  # Those values are changed upon logging
                elif name == 'cs':
                    # Using repr because of cs-bug
                    # (https://github.com/automl/ConfigSpace/issues/25)
                    self.assertEqual(repr(scen1.cs), repr(scen2.cs))
                elif name == 'feature_dict':
                    self.assertEqual(len(scen1.feature_dict),
                                     len(scen2.feature_dict))
                    for key in scen1.feature_dict:
                        self.assertTrue((scen1.feature_dict[key] ==
                                         scen2.feature_dict[key]).all())
                else:
                    self.assertEqual(getattr(scen1, name),
                                     getattr(scen2, name))

        # First check with file-paths defined
        self.test_scenario_dict['feature_file'] = 'test/test_files/scenario_test/features_multiple.txt'
        scenario = Scenario(self.test_scenario_dict)
        scenario.write()
        path = os.path.join(scenario.output_dir, 'scenario.txt')
        scenario_reloaded = Scenario(path)
        check_scen_eq(scenario, scenario_reloaded)

        # Now create new scenario without filepaths
        self.test_scenario_dict.update({
            'paramfile': None, 'cs': scenario.cs,
            'feature_file': None, 'features': scenario.feature_dict,
            'instance_file': None, 'instances': scenario.train_insts,
            'test_instance_file': None, 'test_instances': scenario.test_insts})
        scenario_no_fn = Scenario(self.test_scenario_dict)
        scenario_reloaded = Scenario(path)
        check_scen_eq(scenario_no_fn, scenario_reloaded)

    @mock.patch.object(os, 'makedirs')
    @mock.patch.object(os.path, 'isdir')
    def test_write_except(self, patch_isdir, patch_mkdirs):
        patch_isdir.return_value = False
        patch_mkdirs.side_effect = OSError()
        with self.assertRaises(OSError) as cm:
            scen = Scenario(self.test_scenario_dict)
            scen.write()

    def test_no_output_dir(self):
        self.test_scenario_dict['output_dir'] = ""
        scenario = Scenario(self.test_scenario_dict)
        scenario.write()
        self.assertFalse(scenario.out_writer.write_scenario_file(scenario))

    def test_par_factor(self):
        # Test setting the default value of 1 if no factor is given
        scenario_dict = self.test_scenario_dict
        scenario_dict['overall_obj'] = 'mean'
        scenario = Scenario(scenario_dict)
        self.assertEqual(scenario.par_factor, 1)
        scenario_dict['overall_obj'] = 'par'
        scenario = Scenario(scenario_dict)
        self.assertEqual(scenario.par_factor, 1)

    def test_truth_value(self):
        self.assertTrue(_is_truthy("1"))
        self.assertTrue(_is_truthy("true"))
        self.assertTrue(_is_truthy(True))
        self.assertFalse(_is_truthy("0"))
        self.assertFalse(_is_truthy("false"))
        self.assertFalse(_is_truthy(False))
        self.assertRaises(ValueError, _is_truthy, "something")

    def test_str_cast_instances(self):
        self.scen = Scenario({'cs': None,
                              'instances': [[1], [2]],
                              'run_obj': 'quality'})
        self.assertIsInstance(self.scen.train_insts[0], str)
        self.assertIsInstance(self.scen.train_insts[1], str)


if __name__ == "__main__":
    unittest.main()
