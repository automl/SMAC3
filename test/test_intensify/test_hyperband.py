import unittest
from nose.plugins.attrib import attr

import logging
import numpy as np
import time

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.intensification.hyperband import Hyperband
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.tae.execute_ta_run import StatusType
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger


def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='a',
                                                       lower=0,
                                                       upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='b',
                                                       lower=0,
                                                       upper=100))
    return cs


class TestHyperband(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.rh = RunHistory(aggregate_func=average_cost)
        self.cs = get_config_space()
        self.config1 = Configuration(self.cs,
                                     values={'a': 0, 'b': 100})
        self.config2 = Configuration(self.cs,
                                     values={'a': 100, 'b': 0})
        self.config3 = Configuration(self.cs,
                                     values={'a': 100, 'b': 100})

        self.scen = Scenario({"cutoff_time": 2, 'cs': self.cs,
                              "run_obj": 'runtime',
                              "output_dir": ''})
        self.stats = Stats(scenario=self.scen)
        self.stats.start_timing()

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

    @attr('slow')
    def test_intensify_1(self):
        """
           test intensify for incumbent
        """

        def target(x):
            return (x['a'] + 1) / 1000.

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = Hyperband(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[1], min_budget=0.1, max_budget=1, eta=2)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=0,
                    additional_info=None)

        inc, _ = intensifier.intensify(challengers=[self.config2],
                                       incumbent=self.config1,
                                       run_history=self.rh,
                                       aggregate_func=average_cost)

        self.assertEqual(intensifier.s_max, 3)
        self.assertEqual(inc, self.config1)

    @attr('slow')
    def test_intensify_2(self):
        """
           test intensify for challengers used in iterations
        """

        def target(x):
            return (2*x['a'] + x['b'] + 1) / 1000.

        stats = Stats(scenario=self.scen)
        stats.start_timing()
        taf = ExecuteTAFuncDict(ta=target, stats=stats, run_obj="quality")
        taf.runhistory = RunHistory(aggregate_func=average_cost)
        taf.runhistory.overwrite_existing_runs = True

        intensifier = Hyperband(
            tae_runner=taf, stats=taf.stats,
            traj_logger=TrajLogger(output_dir=None, stats=taf.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[1], min_budget=0.5, max_budget=1, eta=2)

        # ensuring correct parameter initialization
        self.assertEqual(intensifier.s, intensifier.s_max)
        self.assertEqual(intensifier.s_max, 1)
        self.assertEqual(intensifier.hb_iters, 0)

        # 1st hyperband run - should run only 2 configurations
        challengers = [self.config1, self.config3, self.config2]
        inc, _ = intensifier.intensify(challengers=challengers,
                                       incumbent=None,
                                       run_history=taf.runhistory,
                                       aggregate_func=average_cost)
        # check configuration runs - should have only 1 SH run with 2 configs
        # 3 runs (0.5 -> 2 runs, 1 -> 1 run)
        self.assertEqual(intensifier.s, intensifier.s_max-1)
        self.assertEqual(taf.stats.ta_runs, 3)
        self.assertEqual(taf.stats.n_configs, 2)

        # 2nd hyperband run - should run only 1 configuration
        challengers = [self.config2, self.config1, self.config3]
        inc, _ = intensifier.intensify(challengers=challengers,
                                       incumbent=inc,
                                       run_history=taf.runhistory,
                                       aggregate_func=average_cost)
        # check configuration runs - should have only 1 SH run with 1 config
        # 1 extra run (1 -> 1 run)
        self.assertEqual(taf.stats.ta_runs, 4)
        self.assertEqual(taf.stats.n_configs, 3)
        # hyperband completes 1 run & s reset to s_max
        self.assertEqual(intensifier.s, intensifier.s_max)
        self.assertEqual(intensifier.hb_iters, 1)
        # correct incumbent selected
        self.assertEqual(inc, self.config1)
