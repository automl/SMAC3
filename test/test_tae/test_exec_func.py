import time
import unittest
import unittest.mock

import numpy as np
import pynisher

from smac.tae.execute_func import ExecuteTAFunc
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.smbo.smbo import StatusType


class TestExecuteFunc(unittest.TestCase):

    def setUp(self):
        self.scenario = Scenario({})
        self.stats = Stats(scenario=self.scenario)

    def test_run(self):
        target = lambda x, _: x**2
        taf = ExecuteTAFunc(func=target, stats=self.stats)
        rval = taf.run(config=2)
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], 4)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())

        target = lambda x, instance, seed: (x ** 2, {'key': seed,
                                                     'instance': instance})
        taf = ExecuteTAFunc(func=target, stats=self.stats)
        rval = taf.run(config=2, instance='test')
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], 4)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], {'key': 12345, 'instance': 'test'})

    def test_for_runtime(self):
        target = lambda x, _: x**2
        taf = ExecuteTAFunc(func=target, stats=self.stats, run_obj='runtime')
        rval = taf.run(config=2)
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], rval[2])
        self.assertEqual(rval[3], dict())

    def test_memout(self):
        def fill_memory(*args):
            a = np.random.random_sample((10000, 10000)).astype(np.float64)
            return np.sum(a)

        taf = ExecuteTAFunc(func=fill_memory, stats=self.stats)
        rval = taf.run(config=None, memory_limit=1024)
        self.assertEqual(rval[0], StatusType.MEMOUT)
        self.assertEqual(rval[1], 1234567890)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())

    def test_timeout(self):
        def run_over_time(*args):
            time.sleep(5)

        taf = ExecuteTAFunc(func=run_over_time, stats=self.stats)
        rval = taf.run(config=None, cutoff=1)
        self.assertEqual(rval[0], StatusType.TIMEOUT)
        self.assertEqual(rval[1], 1234567890)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())

    def test_timeout_runtime(self):
        def run_over_time(*args):
            time.sleep(5)

        taf = ExecuteTAFunc(func=run_over_time, stats=self.stats,
                            run_obj='runtime', par_factor=11)
        rval = taf.run(config=None, cutoff=1)
        self.assertEqual(rval[0], StatusType.TIMEOUT)
        self.assertGreaterEqual(rval[1], 11)
        self.assertGreaterEqual(rval[2], 1)
        self.assertEqual(rval[3], dict())

    @unittest.mock.patch('autosklearn.evaluation.eval_holdout')
    def test_fail_silent(self, pynisher_mock):
        pynisher_mock.return_value = None

        taf = ExecuteTAFunc(func=pynisher_mock, stats=self.stats)
        rval = taf.run(config=None, cutoff=1)
        self.assertEqual(rval[0], StatusType.CRASHED)
        self.assertEqual(rval[1], 1234567890)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())
