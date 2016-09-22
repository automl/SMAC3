import unittest
from unittest import mock

from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.scenario.scenario import Scenario
from smac.smbo.smbo import StatusType


class TestExecuteTARun(unittest.TestCase):

    @mock.patch.object(ExecuteTARun, 'run', autospec=True)
    def test_memory_limit_usage(self, run_mock):
        run_mock.return_value = StatusType.SUCCESS, 12345.0, 1.2345, {}

        stats = Stats(Scenario({}))
        stats.start_timing()
        tae = ExecuteTARun(lambda x : x**2, stats, run_obj='quality')

        self.assertRaisesRegex(ValueError, 'Target algorithm executor '
                                           'ExecuteTARun does not support restricting the memory usage.',
                               tae.start, {'x': 2}, 'a', memory_limit=123)
        tae._supports_memory_limit = True

        rval = tae.start({'x': 2}, 'a', memory_limit=10)
        self.assertEqual(rval, run_mock.return_value)
