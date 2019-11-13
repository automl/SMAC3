'''
Created on Apr 25, 2015

@author: Andre Biedenkapp
'''
import unittest
import logging
import json
import os

try:
    import unittest.mock
    from unittest.mock import patch
except:
    from mock import patch
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.io.traj_logging import TrajEntry

from smac.configspace import ConfigurationSpace, Configuration, CategoricalHyperparameter, Constant
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats

class TrajLoggerTest(unittest.TestCase):

    def mocked_get_used_wallclock_time(self):
        self.value += 1
        return self.value

    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.value = 0
        self.cs = ConfigurationSpace()

    def test_init(self):
        scen = Scenario(scenario={'run_obj': 'quality', 'cs': self.cs,
                                  'output_dir': ''})
        stats = Stats(scen)
        TrajLogger(output_dir='./tmp_test_folder', stats=stats)
        self.assertFalse(os.path.exists('smac3-output'))
        self.assertTrue(os.path.exists('tmp_test_folder'))

    def test_oserror(self):
        scen = Scenario(scenario={'run_obj': 'quality', 'cs': self.cs,
                                  'output_dir': ''})
        stats = Stats(scen)
        # test OSError
        with patch('os.makedirs') as osMock:
            osMock.side_effect = OSError()
            self.assertRaises(OSError, TrajLogger, output_dir='./tmp_test_folder', stats=stats)

    @patch('smac.stats.stats.Stats')
    def test_add_entry(self, mock_stats):

        tl = TrajLogger(output_dir='./tmp_test_folder', stats=mock_stats)
        test_config = {'param_a': 0.5,
                       'param_b': 1,
                       'param_c': 'value'}

        mock_stats.ta_time_used = .5
        mock_stats.get_used_wallclock_time = self.mocked_get_used_wallclock_time
        mock_stats.ta_runs = 1

        tl.add_entry(0.9, 1, test_config)

        self.assertTrue(os.path.exists('tmp_test_folder/traj_old.csv'))
        self.assertTrue(os.path.exists('tmp_test_folder/traj_aclib2.json'))
        self.assertTrue(os.path.exists('tmp_test_folder/traj_alljson.json'))

        with open('tmp_test_folder/traj_old.csv') as to:
            data = to.read().split('\n')[:-1]

        header = data[0].split(',')
        self.assertEquals(header[0], '"CPU Time Used"')
        self.assertEquals(header[-1], '"Configuration..."')

        data = list(map(lambda x: x.split(', '), data[1:]))
        frmt_str = '%1.6f'
        self.assertEquals(frmt_str % .5, data[0][0])
        self.assertEquals(frmt_str % .9, data[0][1])
        self.assertEquals(frmt_str % 0.5, data[0][-4])

        with open('tmp_test_folder/traj_aclib2.json') as js:
            json_dict = json.load(js)

        self.assertEquals(json_dict['cpu_time'], .5)
        self.assertEquals(json_dict['cost'], 0.9)
        self.assertEquals(len(json_dict['incumbent']), 3)
        self.assertTrue("param_a='0.5'" in json_dict['incumbent'])

        with open('tmp_test_folder/traj_alljson.json') as js:
            json_dict = json.load(js)

        self.assertEquals(json_dict['cpu_time'], .5)
        self.assertEquals(json_dict['cost'], 0.9)
        self.assertEquals(len(json_dict['incumbent']), 3)
        self.assertTrue(json_dict["incumbent"]["param_a"] == 0.5)

        # And finally, test the list that's added to the trajectory class
        self.assertEqual(tl.trajectory[0], TrajEntry(0.9, 1,
                                            {'param_c': 'value', 'param_b': 1,
                                             'param_a': 0.5}, 1, 0.5, 1))
        # Test named-tuple-access:
        self.assertEqual(tl.trajectory[0].train_perf, 0.9)
        self.assertEqual(tl.trajectory[0].incumbent_id, 1)
        self.assertEqual(tl.trajectory[0].ta_runs, 1)
        self.assertEqual(tl.trajectory[0].ta_time_used, 0.5)
        self.assertEqual(tl.trajectory[0].wallclock_time, 1)
        self.assertEqual(len(tl.trajectory), 1)

    @patch('smac.stats.stats.Stats')
    def test_add_multiple_entries(self, mock_stats):
        tl = TrajLogger(output_dir='./tmp_test_folder', stats=mock_stats)

        test_config = {'param_a': 0.5,
                       'param_b': 1,
                       'param_c': 'value'}
        mock_stats.ta_time_used = 0.5
        mock_stats.get_used_wallclock_time = self.mocked_get_used_wallclock_time
        mock_stats.ta_runs = 1
        tl.add_entry(0.9, 1, test_config)

        mock_stats.ta_runs = 2
        mock_stats.ta_time_used = 0
        tl.add_entry(1.3, 1, test_config)

        test_config['param_a'] = 0.
        mock_stats.ta_time_used = 0
        tl.add_entry(0.7, 2, test_config)

        self.assertTrue(os.path.exists('tmp_test_folder/traj_old.csv'))
        self.assertTrue(os.path.exists('tmp_test_folder/traj_aclib2.json'))
        self.assertTrue(os.path.exists('tmp_test_folder/traj_alljson.json'))

        with open('tmp_test_folder/traj_old.csv') as to:
            data = to.read().split('\n')[:-1]

        header = data[0].split(',')
        self.assertEquals(header[0], '"CPU Time Used"')
        self.assertEquals(header[-1], '"Configuration..."')

        data = list(map(lambda x: x.split(', '), data[1:]))
        frmt_str = '%1.6f'
        self.assertEquals(frmt_str % 0.5, data[0][0])
        self.assertEquals(frmt_str % 0.9, data[0][1])
        self.assertEquals(frmt_str % 0.5, data[0][-4])

        self.assertEquals(frmt_str % 0, data[1][0])
        self.assertEquals(frmt_str % 1.3, data[1][1])
        self.assertEquals(frmt_str % 2, data[1][-4])

        self.assertEquals(frmt_str % 0, data[2][0])
        self.assertEquals(frmt_str % .7, data[2][1])
        self.assertEquals(frmt_str % 3, data[2][-4])

        json_dicts = []
        with open('tmp_test_folder/traj_aclib2.json') as js:
            data = js.read().split('\n')[:-1]

        for d in data:
            json_dicts.append(json.loads(d))

        self.assertEquals(json_dicts[0]['cpu_time'], .5)
        self.assertEquals(json_dicts[0]['cost'], 0.9)
        self.assertEquals(len(json_dicts[0]['incumbent']), 3)
        self.assertTrue("param_a='0.5'" in json_dicts[0]['incumbent'])

        with open('tmp_test_folder/traj_alljson.json') as js:
            data = js.read().split('\n')[:-1]
        json_dicts = [json.loads(d) for d in data]

        self.assertEquals(json_dicts[0]['cpu_time'], .5)
        self.assertEquals(json_dicts[0]['cost'], 0.9)
        self.assertEquals(len(json_dicts[0]['incumbent']), 3)
        self.assertTrue(json_dicts[0]["incumbent"]["param_a"] == 0.5)

    @patch('smac.stats.stats.Stats')
    def test_ambigious_categoricals(self, mock_stats):
        tl = TrajLogger(output_dir='./tmp_test_folder', stats=mock_stats)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([
            # Adding another value to the choices with "True", for example, would break the interpretation with aclib2
            CategoricalHyperparameter('ambigous_categorical', choices=[3, 7.2, True, 0, 'random_string']),
            Constant('ambigous_constant', value='False')
        ])
        mock_stats.ta_time_used = 0.5
        mock_stats.get_used_wallclock_time = self.mocked_get_used_wallclock_time
        mock_stats.ta_runs = 1
        tl.add_entry(0.9, 1, Configuration(cs, {'ambigous_categorical' : True,
                                                'ambigous_constant' : 'False'}))

        mock_stats.ta_runs = 2
        mock_stats.ta_time_used = 0
        tl.add_entry(1.3, 1, Configuration(cs, {'ambigous_categorical' : True,
                                                'ambigous_constant' : 'False'}))

        mock_stats.ta_time_used = 0
        tl.add_entry(0.7, 2, Configuration(cs, {'ambigous_categorical' : 7.2,
                                                'ambigous_constant' : 'False'}))

        self.assertTrue(os.path.exists('tmp_test_folder/traj_aclib2.json'))
        self.assertTrue(os.path.exists('tmp_test_folder/traj_alljson.json'))

        from_aclib2 = tl.read_traj_aclib_format('tmp_test_folder/traj_aclib2.json', cs)
        from_alljson = tl.read_traj_alljson_format('tmp_test_folder/traj_alljson.json', cs)

        for reloaded in [from_aclib2, from_alljson]:
            self.assertIsInstance(reloaded[0]['incumbent']['ambigous_categorical'], bool)
            self.assertIsInstance(reloaded[-1]['incumbent']['ambigous_categorical'], float)
            self.assertIsInstance(reloaded[0]['incumbent']['ambigous_constant'], str)
            self.assertIsInstance(reloaded[-1]['incumbent']['ambigous_constant'], str)

        # An example breaking for aclib2 (unfixable) but working with alljson
        bad_cs = ConfigurationSpace()
        bad_cs.add_hyperparameters([
            CategoricalHyperparameter('ambigous_categorical', choices=[3, 7.2, True, "True", 0, 'random_string']),
            Constant('ambigous_constant', value='False')
        ])

        from_aclib2 = tl.read_traj_aclib_format('tmp_test_folder/traj_aclib2.json', bad_cs)
        from_alljson = tl.read_traj_alljson_format('tmp_test_folder/traj_alljson.json', bad_cs)

        # Wrong! but passes:
        self.assertIsInstance(from_aclib2[0]['incumbent']['ambigous_categorical'], str)

        self.assertIsInstance(from_alljson[0]['incumbent']['ambigous_categorical'], bool)
        self.assertIsInstance(from_alljson[-1]['incumbent']['ambigous_categorical'], float)
        self.assertIsInstance(from_alljson[0]['incumbent']['ambigous_constant'], str)
        self.assertIsInstance(from_alljson[-1]['incumbent']['ambigous_constant'], str)

    def tearDown(self):
        if os.path.exists('tmp_test_folder/traj_old.csv'):
            os.remove('tmp_test_folder/traj_old.csv')
        if os.path.exists('tmp_test_folder/traj_aclib2.json'):
            os.remove('tmp_test_folder/traj_aclib2.json')
        if os.path.exists('tmp_test_folder/traj_alljson.json'):
            os.remove('tmp_test_folder/traj_alljson.json')
        if os.path.exists('tmp_test_folder'):
            os.rmdir('tmp_test_folder')

if __name__ == "__main__":
    unittest.main()
