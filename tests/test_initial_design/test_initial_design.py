import unittest
import unittest.mock

import numpy as np
from ConfigSpace import Configuration, UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.initial_design.initial_design import InitialDesign
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.tae.execute_func import ExecuteTAFuncDict

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestSingleInitialDesign(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.cs.add_hyperparameter(UniformFloatHyperparameter(
            name="x1", lower=1, upper=10, default_value=1)
        )
        self.scenario = Scenario({
            'cs': self.cs,
            'run_obj': 'quality',
            'output_dir': '',
            'ta_run_limit': 100,
        })
        self.stats = Stats(scenario=self.scenario)
        self.rh = RunHistory()
        self.ta = ExecuteTAFuncDict(lambda x: x["x1"]**2, stats=self.stats)

    def test_single_default_config_design(self):
        self.stats.start_timing()
        tj = TrajLogger(output_dir=None, stats=self.stats)

        dc = DefaultConfiguration(
            cs=self.cs,
            traj_logger=tj,
            rng=np.random.RandomState(seed=12345),
            ta_run_limit=self.scenario.ta_run_limit
        )

        # should return only the default config
        configs = dc.select_configurations()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]['x1'], 1)

    def test_multi_config_design(self):
        self.stats.start_timing()
        tj = TrajLogger(output_dir=None, stats=self.stats)
        _ = np.random.RandomState(seed=12345)

        configs = [Configuration(configuration_space=self.cs, values={"x1": 4}),
                   Configuration(configuration_space=self.cs, values={"x1": 2})]
        dc = InitialDesign(
            cs=self.cs,
            traj_logger=tj,
            rng=np.random.RandomState(seed=12345),
            ta_run_limit=self.scenario.ta_run_limit,
            configs=configs
        )

        # selects multiple initial configurations to run
        # since the configs were passed to initial design, it should return the same
        init_configs = dc.select_configurations()
        self.assertEqual(len(init_configs), 2)
        self.assertEqual(init_configs, configs)

    def test_init_budget(self):
        self.stats.start_timing()
        tj = TrajLogger(output_dir=None, stats=self.stats)
        _ = np.random.RandomState(seed=12345)

        kwargs = dict(
            cs=self.cs,
            traj_logger=tj,
            rng=np.random.RandomState(seed=12345),
            ta_run_limit=self.scenario.ta_run_limit
        )

        configs = [Configuration(configuration_space=self.cs, values={"x1": 4}),
                   Configuration(configuration_space=self.cs, values={"x1": 2})]
        dc = InitialDesign(
            configs=configs,
            init_budget=3,
            **kwargs,
        )
        self.assertEqual(dc.init_budget, 3)

        dc = InitialDesign(
            init_budget=3,
            **kwargs,
        )
        self.assertEqual(dc.init_budget, 3)

        configs = [Configuration(configuration_space=self.cs, values={"x1": 4}),
                   Configuration(configuration_space=self.cs, values={"x1": 2})]
        dc = InitialDesign(
            configs=configs,
            **kwargs,
        )
        self.assertEqual(dc.init_budget, 2)

        dc = InitialDesign(
            **kwargs,
        )
        self.assertEqual(dc.init_budget, 10)

        with self.assertRaisesRegex(
            ValueError,
            'Initial budget 200 cannot be higher than the run limit 100.',
        ):
            InitialDesign(init_budget=200, **kwargs)

        with self.assertRaisesRegex(
            ValueError,
            'Need to provide either argument `init_budget`, `configs` or `n_configs_x_params`, '
            'but provided none of them.',
        ):
            InitialDesign(**kwargs, n_configs_x_params=None)

    def test__select_configurations(self):
        kwargs = dict(
            cs=self.cs,
            rng=np.random.RandomState(1),
            traj_logger=unittest.mock.Mock(),
            ta_run_limit=1000,
            configs=None,
            n_configs_x_params=None,
            max_config_fracs=0.25,
            init_budget=1,
        )
        init_design = InitialDesign(**kwargs)
        with self.assertRaises(NotImplementedError):
            init_design._select_configurations()
