import time
import unittest

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.cli.scenario import Scenario
from smac.cli.traj_logging import TrajLogger
from smac.intensification.abstract_racer import RunInfoIntent
from smac.intensification.simple_intensifier import SimpleIntensifier
from smac.runhistory.runhistory import RunHistory, RunInfo, TrialValue
from smac.stats import Stats
from smac.runner.abstract_runner import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name="a", lower=0, upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name="b", lower=0, upper=100))
    return cs


def target_from_run_info(RunInfo):
    value_from_config = sum([a for a in RunInfo.config.get_dictionary().values()])
    return TrialValue(
        cost=value_from_config,
        time=0.5,
        status=StatusType.SUCCESS,
        starttime=time.time(),
        endtime=time.time() + 1,
        additional_info={},
    )


class TestSimpleIntensifier(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)

        self.rh = RunHistory()
        self.cs = get_config_space()
        self.config1 = Configuration(self.cs, values={"a": 7, "b": 11})
        self.config2 = Configuration(self.cs, values={"a": 13, "b": 17})
        self.config3 = Configuration(self.cs, values={"a": 0, "b": 7})
        self.config4 = Configuration(self.cs, values={"a": 29, "b": 31})

        self.scen = Scenario({"cutoff_time": 2, "cs": self.cs, "run_obj": "runtime", "output_dir": ""})
        self.stats = Stats(scenario=self.scen)
        self.stats.start_timing()

        # Create the base object
        self.intensifier = SimpleIntensifier(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            deterministic=True,
            run_obj_time=False,
            instances=[1],
        )

    def test_get_next_run(self):
        """
        Makes sure that sampling a configuration returns a valid
        configuration
        """
        intent, run_info = self.intensifier.get_next_run(
            challengers=[self.config1],
            incumbent=None,
            run_history=self.rh,
            n_workers=1,
            chooser=None,
        )

        self.assertEqual(intent, RunInfoIntent.RUN)

        self.assertEqual(
            run_info,
            RunInfo(
                config=self.config1,
                instance=1,
                instance_specific="0",
                seed=0,
                cutoff=None,
                capped=False,
                budget=0.0,
            ),
        )

    def test_get_next_run_waits_if_no_workers(self):
        """
        In the case all workers are busy, we wait so that we do
        not saturate the process with configurations that will not
        finish in time
        """
        intent, run_info = self.intensifier.get_next_run(
            challengers=[self.config1, self.config2],
            incumbent=None,
            run_history=self.rh,
            n_workers=1,
            chooser=None,
        )

        # We can get the configuration 1
        self.assertEqual(intent, RunInfoIntent.RUN)
        self.assertEqual(
            run_info,
            RunInfo(
                config=self.config1,
                instance=1,
                instance_specific="0",
                seed=0,
                cutoff=None,
                capped=False,
                budget=0.0,
            ),
        )

        # We should not get configuration 2
        # As there is just 1 worker
        intent, run_info = self.intensifier.get_next_run(
            challengers=[self.config2],
            incumbent=None,
            run_history=self.rh,
            n_workers=1,
            chooser=None,
        )
        self.assertEqual(intent, RunInfoIntent.WAIT)
        self.assertEqual(
            run_info,
            RunInfo(
                config=None,
                instance=None,
                instance_specific="0",
                seed=0,
                cutoff=None,
                capped=False,
                budget=0.0,
            ),
        )

    def test_process_results(self):
        """
        Makes sure that we can process the results of a completed
        configuration
        """
        intent, run_info = self.intensifier.get_next_run(
            challengers=[self.config1, self.config2],
            incumbent=None,
            run_history=self.rh,
            n_workers=1,
            chooser=None,
        )
        result = TrialValue(
            cost=1,
            time=0.5,
            status=StatusType.SUCCESS,
            starttime=1,
            endtime=2,
            additional_info=None,
        )
        self.rh.add(
            config=run_info.config,
            cost=1,
            time=0.5,
            status=StatusType.SUCCESS,
            instance_id=run_info.instance,
            seed=run_info.seed,
            additional_info=None,
        )

        incumbent, inc_perf = self.intensifier.process_results(
            run_info=run_info,
            incumbent=None,
            run_history=self.rh,
            time_bound=np.inf,
            result=result,
        )
        self.assertEqual(incumbent, run_info.config)
        self.assertEqual(inc_perf, 1)


if __name__ == "__main__":
    unittest.main()
