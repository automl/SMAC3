import time
import unittest
import unittest.mock

from smac.configspace import ConfigurationSpace
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae import StatusType
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.tae.serial_runner import SerialRunner

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def target(x, seed, instance):
    return x ** 2, {'key': seed, 'instance': instance}


def target_delayed(x, seed, instance):
    time.sleep(1)
    return x ** 2, {'key': seed, 'instance': instance}


class TestSerialRunner(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({'cs': self.cs,
                                  'run_obj': 'quality',
                                  'output_dir': ''})
        self.stats = Stats(scenario=self.scenario)

    def test_run(self):
        """Makes sure that we are able to run a configuration and
        return the expected values/types"""

        # We use the funcdict as a mechanism to test SerialRunner
        runner = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj='quality')
        self.assertIsInstance(runner, SerialRunner)

        run_info = RunInfo(config=2, instance='test', instance_specific="0",
                           seed=0, cutoff=None, capped=False, budget=0.0)

        # submit runs! then get the value
        runner.submit_run(run_info)
        run_values = runner.get_finished_runs()
        self.assertEqual(len(run_values), 1)
        self.assertIsInstance(run_values, list)
        self.assertIsInstance(run_values[0][0], RunInfo)
        self.assertIsInstance(run_values[0][1], RunValue)
        self.assertEqual(run_values[0][1].cost, 4)
        self.assertEqual(run_values[0][1].status, StatusType.SUCCESS)

    def test_serial_runs(self):

        # We use the funcdict as a mechanism to test SerialRunner
        runner = ExecuteTAFuncDict(ta=target_delayed, stats=self.stats, run_obj='quality')
        self.assertIsInstance(runner, SerialRunner)

        run_info = RunInfo(config=2, instance='test', instance_specific="0",
                           seed=0, cutoff=None, capped=False, budget=0.0)
        runner.submit_run(run_info)
        run_info = RunInfo(config=3, instance='test', instance_specific="0",
                           seed=0, cutoff=None, capped=False, budget=0.0)
        runner.submit_run(run_info)
        run_values = runner.get_finished_runs()
        self.assertEqual(len(run_values), 2)

        # To make sure runs launched serially, we just make sure that the end time of
        # a run is later than the other
        # Results are returned in left to right
        self.assertLessEqual(int(run_values[1][1].endtime), int(run_values[0][1].starttime))

        # No wait time in serial runs!
        start = time.time()
        runner.wait()

        # The run takes a second, so 0.5 is sufficient
        self.assertLess(time.time() - start, 0.5)
        pass


if __name__ == "__main__":
    unittest.main()
