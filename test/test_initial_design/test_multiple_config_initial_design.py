import unittest

import numpy as np

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace, Configuration

from smac.runhistory.runhistory import RunHistory
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.initial_design.multi_config_initial_design import MultiConfigInitialDesign
from smac.utils.io.traj_logging import TrajLogger
from smac.optimizer.objective import average_cost
from smac.intensification.intensification import Intensifier

class TestMultiInitialDesign(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameter(UniformFloatHyperparameter(
            name="x1", lower=1, upper=10, default_value=2)
        )
        self.scenario = Scenario({'cs': self.cs, 'run_obj': 'quality',
                                  'output_dir': ''})
        self.ta = ExecuteTAFuncDict(lambda x: x["x1"]**2)

    def test_multi_config_design(self):
        stats = Stats(scenario=self.scenario)
        stats.start_timing()
        self.ta.stats = stats
        tj = TrajLogger(output_dir=None, stats=stats)
        rh = RunHistory(aggregate_func=average_cost)
        self.ta.runhistory = rh
        rng = np.random.RandomState(seed=12345)

        intensifier = Intensifier(tae_runner=self.ta, stats=stats, traj_logger=tj, rng=rng, instances=[None],
                                  run_obj_time=False)

        configs = [Configuration(configuration_space=self.cs, values={"x1":4}),
                   Configuration(configuration_space=self.cs, values={"x1":2})]
        dc = MultiConfigInitialDesign(tae_runner=self.ta, scenario=self.scenario, stats=stats,
                             traj_logger=tj, runhistory=rh, rng=rng, configs=configs,
                             intensifier=intensifier, aggregate_func=average_cost)

        inc = dc.run()
        self.assertTrue(stats.ta_runs==4)  # two runs per config
        self.assertTrue(len(rh.data)==4)  # two runs per config
        self.assertTrue(rh.get_cost(inc) == 4)

