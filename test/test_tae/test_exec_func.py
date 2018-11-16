import os
import sys
import time
import unittest
import unittest.mock
from nose.plugins.attrib import attr

import numpy as np

from smac.configspace import ConfigurationSpace, Configuration
from smac.tae.execute_func import ExecuteTAFuncDict, ExecuteTAFuncArray
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import StatusType

class TestExecuteFunc(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({'cs': self.cs,
                                  'run_obj': 'quality',
                                  'output_dir': ''})
        self.stats = Stats(scenario=self.scenario)

    def test_run(self):
        target = lambda x: x**2
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        rval = taf.run(config=2)
        self.assertFalse(taf._accepts_instance)
        self.assertFalse(taf._accepts_seed)
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], 4)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())

        target = lambda x, seed: (x ** 2, {'key': seed})
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        rval = taf.run(config=2, instance='test')
        self.assertFalse(taf._accepts_instance)
        self.assertTrue(taf._accepts_seed)
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], 4)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], {'key': 12345})

        target = lambda x, seed, instance: (x ** 2, {'key': seed,
                                                     'instance': instance})
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        rval = taf.run(config=2, instance='test')
        self.assertTrue(taf._accepts_instance)
        self.assertTrue(taf._accepts_seed)
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], 4)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], {'key': 12345, 'instance': 'test'})
        
    
    def test_run_wo_pynisher(self):
        target = lambda x: x**2
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, use_pynisher=False)
        rval = taf.run(config=2)
        self.assertFalse(taf._accepts_instance)
        self.assertFalse(taf._accepts_seed)
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], 4)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())

        target = lambda x: None
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, use_pynisher=False)
        rval = taf.run(config=2)
        self.assertFalse(taf._accepts_instance)
        self.assertFalse(taf._accepts_seed)
        self.assertEqual(rval[0], StatusType.CRASHED)
        self.assertEqual(rval[1], 2147483647.0)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())


    @unittest.mock.patch.object(Configuration, 'get_dictionary')
    def test_run_execute_func_for_fmin(self, mock):
        mock.return_value = {'x1': 2, 'x2': 1}
        c = Configuration(configuration_space=self.cs, values={})
        target = lambda x: x[0] ** 2 + x[1]
        taf = ExecuteTAFuncArray(target, stats=self.stats)
        rval = taf._call_ta(target, c)
        self.assertEqual(rval, 5)

    def test_memout(self):
        def fill_memory(*args):
            a = np.random.random_sample((10000, 10000)).astype(np.float64)
            return np.sum(a)

        taf = ExecuteTAFuncDict(ta=fill_memory, stats=self.stats,
                                memory_limit=1024)
        rval = taf.run(config=None)

        platform = os.getenv('TRAVIS_OS_NAME')
        if platform is None:
            platform = {'linux': 'linux',
                        'darwin': 'osx'}.get(sys.platform)

        print(platform, sys.platform)
        if platform == 'linux':
            self.assertEqual(rval[0], StatusType.MEMOUT)
            self.assertEqual(rval[1], 2147483647.0)
            self.assertGreaterEqual(rval[2], 0.0)
            self.assertEqual(rval[3], dict())
        elif platform == 'osx':
            self.assertEqual(rval[0], StatusType.SUCCESS)
        else:
            raise ValueError('Test cannot be performed on platform %s' %
                             sys.platform)

    @attr('slow')
    def test_timeout(self):
        def run_over_time(*args):
            time.sleep(5)

        taf = ExecuteTAFuncDict(ta=run_over_time, stats=self.stats)
        rval = taf.run(config=None, cutoff=1)
        self.assertEqual(rval[0], StatusType.TIMEOUT)
        self.assertEqual(rval[1], 2147483647.0)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())

    @attr('slow')
    def test_timeout_runtime(self):
        def run_over_time(*args):
            time.sleep(5)

        taf = ExecuteTAFuncDict(ta=run_over_time, stats=self.stats,
                                run_obj='runtime', par_factor=11)
        rval = taf.run(config=None, cutoff=1)
        self.assertEqual(rval[0], StatusType.TIMEOUT)
        self.assertGreaterEqual(rval[1], 11)
        self.assertGreaterEqual(rval[2], 1)
        self.assertEqual(rval[3], dict())

    def test_fail_silent(self):
        def function(*args):
            return

        taf = ExecuteTAFuncDict(ta=function, stats=self.stats)
        rval = taf.run(config=None, cutoff=1)
        self.assertEqual(rval[0], StatusType.CRASHED)
        self.assertEqual(rval[1], 2147483647.0)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())

    def test_cutoff_too_large(self):
        target = lambda x: x**2
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        self.assertRaises(ValueError, taf.run, config=2, cutoff=65536)

