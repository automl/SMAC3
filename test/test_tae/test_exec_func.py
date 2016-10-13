import time
import unittest
import unittest.mock

import numpy as np

from smac.configspace import ConfigurationSpace, Configuration
from smac.tae.execute_func import ExecuteTAFunc, ExecuteTAFunc4FMIN
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import StatusType


class TestExecuteFunc(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({'cs': self.cs})
        self.stats = Stats(scenario=self.scenario)

    def test_run(self):
        target = lambda x, _: x**2
        taf = ExecuteTAFunc(ta=target, stats=self.stats)
        rval = taf.run(config=2)
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], 4)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())

        target = lambda x, instance, seed: (x ** 2, {'key': seed,
                                                     'instance': instance})
        taf = ExecuteTAFunc(ta=target, stats=self.stats)
        rval = taf.run(config=2, instance='test')
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], 4)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], {'key': 12345, 'instance': 'test'})

    @unittest.mock.patch.object(Configuration, 'get_dictionary')
    def test_run_execute_func_for_fmin(self, mock):
        mock.return_value = {'x1': 2, 'x2': 1}
        c = Configuration(configuration_space=self.cs, values={})
        target = lambda x: x[0] ** 2 + x[1]
        taf = ExecuteTAFunc4FMIN(target, stats=self.stats)
        rval = taf._call_ta(target, c, None, 1)
        self.assertEqual(rval, (5, {}))

    def test_memout(self):
        def fill_memory(*args):
            a = np.random.random_sample((10000, 10000)).astype(np.float64)
            return np.sum(a)

        taf = ExecuteTAFunc(ta=fill_memory, stats=self.stats)
        rval = taf.run(config=None, memory_limit=1024)
        self.assertEqual(rval[0], StatusType.MEMOUT)
        self.assertEqual(rval[1], 1234567890)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())

    def test_timeout(self):
        def run_over_time(*args):
            time.sleep(5)

        taf = ExecuteTAFunc(ta=run_over_time, stats=self.stats)
        rval = taf.run(config=None, cutoff=1)
        self.assertEqual(rval[0], StatusType.TIMEOUT)
        self.assertEqual(rval[1], 1234567890)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())

    def test_timeout_runtime(self):
        def run_over_time(*args):
            time.sleep(5)

        taf = ExecuteTAFunc(ta=run_over_time, stats=self.stats,
                            run_obj='runtime', par_factor=11)
        rval = taf.run(config=None, cutoff=1)
        self.assertEqual(rval[0], StatusType.TIMEOUT)
        self.assertGreaterEqual(rval[1], 11)
        self.assertGreaterEqual(rval[2], 1)
        self.assertEqual(rval[3], dict())

    def test_fail_silent(self):
        def function(*args):
            return

        taf = ExecuteTAFunc(ta=function, stats=self.stats)
        rval = taf.run(config=None, cutoff=1)
        self.assertEqual(rval[0], StatusType.CRASHED)
        self.assertEqual(rval[1], 1234567890)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())
