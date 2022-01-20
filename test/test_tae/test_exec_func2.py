import os
import sys
import time
import unittest
import unittest.mock

import numpy as np

from smac.configspace import ConfigurationSpace, Configuration
from smac.tae.execute_func import ExecuteTAFuncDict, ExecuteTAFuncArray
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestExecuteFunc(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({'cs': self.cs,
                                  'run_obj': 'quality',
                                  'output_dir': '',
                                  'limit_resources': False})
        self.stats = Stats(scenario=self.scenario)

    def test_run(self):
        def target(x):
            return x**2
        
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        rval = taf.run(config=2)
        
        print(rval)
        
        self.assertFalse(taf._accepts_instance)
        self.assertFalse(taf._accepts_seed)
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], 4)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())

        def target(x, seed):
            return x ** 2, {'key': seed}
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        rval = taf.run(config=2, instance='test')
        self.assertFalse(taf._accepts_instance)
        self.assertTrue(taf._accepts_seed)
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], 4)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], {'key': 12345})

        def target(x, seed, instance):
            return x ** 2, {'key': seed, 'instance': instance}
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        rval = taf.run(config=2, instance='test')
        self.assertTrue(taf._accepts_instance)
        self.assertTrue(taf._accepts_seed)
        self.assertEqual(rval[0], StatusType.SUCCESS)
        self.assertEqual(rval[1], 4)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], {'key': 12345, 'instance': 'test'})

        def target(x):
            raise Exception(x)
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        rval = taf.run(config=2)
        self.assertFalse(taf._accepts_instance)
        self.assertFalse(taf._accepts_seed)
        self.assertEqual(rval[0], StatusType.CRASHED)
        self.assertEqual(rval[1], 2147483647.0)
        self.assertGreaterEqual(rval[2], 0.0)
        self.assertEqual(rval[3], dict())


if __name__ == "__main__":
    t = TestExecuteFunc()
    t.setUp()
    t.test_run()