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

from smac.configspace import ConfigurationSpace
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

    def tearDown(self):
        if os.path.exists('tmp_test_folder/traj_old.csv'):
            os.remove('tmp_test_folder/traj_old.csv')
        if os.path.exists('tmp_test_folder/traj_aclib2.json'):
            os.remove('tmp_test_folder/traj_aclib2.json')
        if os.path.exists('tmp_test_folder'):
            os.rmdir('tmp_test_folder')

if __name__ == "__main__":
    unittest.main()
