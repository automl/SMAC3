import tempfile
import logging
import json
import os

import unittest.mock
from unittest.mock import patch

from smac.utils.io.traj_logging import TrajLogger
from smac.utils.io.traj_logging import TrajEntry

from smac.configspace import ConfigurationSpace,\
    Configuration, CategoricalHyperparameter, Constant, UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


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
        self.cs.add_hyperparameters([
            UniformFloatHyperparameter('param_a', -0.2, 1.77, 1.1),
            UniformIntegerHyperparameter('param_b', -3, 10, 1),
            Constant('param_c', 'value'),
            CategoricalHyperparameter('ambigous_categorical', choices=['True', True, 5]),  # True is ambigous here
        ])
        self.test_config = Configuration(self.cs, {'param_a': 0.5,
                                                   'param_b': 1,
                                                   'param_c': 'value',
                                                   'ambigous_categorical': 5})

    def test_init(self):
        scen = Scenario(scenario={'run_obj': 'quality', 'cs': self.cs, 'output_dir': ''})
        stats = Stats(scen)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'tmp_test_folder')
            TrajLogger(output_dir=path, stats=stats)
            self.assertTrue(os.path.exists(path))

    def test_oserror(self):
        scen = Scenario(scenario={'run_obj': 'quality', 'cs': self.cs, 'output_dir': ''})
        stats = Stats(scen)
        # test OSError
        with patch('os.makedirs') as osMock:
            osMock.side_effect = OSError()
            self.assertRaises(OSError, TrajLogger, output_dir='random_directory', stats=stats)

    @patch('smac.stats.stats.Stats')
    def test_add_entries(self, mock_stats):
        # Mock stats
        mock_stats.ta_time_used = .5
        mock_stats.get_used_wallclock_time = self.mocked_get_used_wallclock_time
        mock_stats.finished_ta_runs = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            tl = TrajLogger(output_dir=tmpdir, stats=mock_stats)

            # Add some entries
            tl.add_entry(0.9, 1, self.test_config, 0)
            mock_stats.ta_runs = 2
            mock_stats.ta_time_used = 0
            tl.add_entry(1.3, 1, self.test_config, 10)
            mock_stats.ta_time_used = 0
            tl.add_entry(0.7, 2, Configuration(self.cs, dict(self.test_config.get_dictionary(), **{'param_a': 0.})), 10)

            # Test the list that's added to the trajectory class
            self.assertEqual(tl.trajectory[0], TrajEntry(0.9, 1, self.test_config, 1, 0.5, 1, 0))
            # Test named-tuple-access:
            self.assertEqual(tl.trajectory[0].train_perf, 0.9)
            self.assertEqual(tl.trajectory[0].incumbent_id, 1)
            self.assertEqual(tl.trajectory[0].ta_runs, 1)
            self.assertEqual(tl.trajectory[0].ta_time_used, 0.5)
            self.assertEqual(tl.trajectory[0].wallclock_time, 1)
            self.assertEqual(tl.trajectory[0].budget, 0)
            self.assertEqual(len(tl.trajectory), 3)

            # Check if the trajectories are generated
            for fn in ['traj_old.csv', 'traj_aclib2.json', 'traj.json']:
                self.assertTrue(os.path.exists(os.path.join(tmpdir, fn)))

            # Load trajectories
            with open(os.path.join(tmpdir, 'traj_old.csv')) as to:
                data = to.read().split('\n')
            with open(os.path.join(tmpdir, 'traj_aclib2.json')) as js_aclib:
                json_dicts_aclib2 = [json.loads(line) for line in js_aclib.read().splitlines()]
            with open(os.path.join(tmpdir, 'traj.json')) as js:
                json_dicts_alljson = [json.loads(line) for line in js.read().splitlines()]

        # Check old format
        header = data[0].split(',')
        self.assertEqual(header[0], '"CPU Time Used"')
        self.assertEqual(header[-1], '"Configuration..."')

        data = list(map(lambda x: x.split(', '), data[1:]))
        frmt_str = '%1.6f'
        self.assertEqual(frmt_str % 0.5, data[0][0])
        self.assertEqual(frmt_str % 0.9, data[0][1])
        self.assertEqual(frmt_str % 0.5, data[0][4])

        self.assertEqual(frmt_str % 0, data[1][0])
        self.assertEqual(frmt_str % 1.3, data[1][1])
        self.assertEqual(frmt_str % 2, data[1][4])

        self.assertEqual(frmt_str % 0, data[2][0])
        self.assertEqual(frmt_str % .7, data[2][1])
        self.assertEqual(frmt_str % 3, data[2][4])

        # Check aclib2-format
        self.assertEqual(json_dicts_aclib2[0]['cpu_time'], .5)
        self.assertEqual(json_dicts_aclib2[0]['cost'], 0.9)
        self.assertEqual(len(json_dicts_aclib2[0]['incumbent']), 4)
        self.assertTrue("param_a='0.5'" in json_dicts_aclib2[0]['incumbent'])
        self.assertTrue("param_a='0.0'" in json_dicts_aclib2[2]['incumbent'])

        # Check alljson-format
        self.assertEqual(json_dicts_alljson[0]['cpu_time'], .5)
        self.assertEqual(json_dicts_alljson[0]['cost'], 0.9)
        self.assertEqual(len(json_dicts_alljson[0]['incumbent']), 4)
        self.assertTrue(json_dicts_alljson[0]["incumbent"]["param_a"] == 0.5)
        self.assertTrue(json_dicts_alljson[2]["incumbent"]["param_a"] == 0.0)
        self.assertEqual(json_dicts_alljson[0]['budget'], 0)
        self.assertEqual(json_dicts_alljson[2]['budget'], 10)

    @patch('smac.stats.stats.Stats')
    def test_ambigious_categoricals(self, mock_stats):
        mock_stats.ta_time_used = 0.5
        mock_stats.get_used_wallclock_time = self.mocked_get_used_wallclock_time
        mock_stats.finished_ta_runs = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            tl = TrajLogger(output_dir=tmpdir, stats=mock_stats)

            problem_config = Configuration(self.cs, {'param_a': 0.0, 'param_b': 2, 'param_c': 'value',
                                                     'ambigous_categorical': True})  # not recoverable without json
            tl.add_entry(0.9, 1, problem_config)

            from_aclib2 = tl.read_traj_aclib_format(os.path.join(tmpdir, 'traj_aclib2.json'), self.cs)
            from_alljson = tl.read_traj_alljson_format(os.path.join(tmpdir, 'traj.json'), self.cs)

        # Wrong! but passes:
        self.assertIsInstance(from_aclib2[0]['incumbent']['ambigous_categorical'], str)
        # Works good for alljson:
        self.assertIsInstance(from_alljson[0]['incumbent']['ambigous_categorical'], bool)


if __name__ == "__main__":
    unittest.main()
