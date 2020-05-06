import unittest

import logging
import numpy as np

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.intensification.hyperband import Hyperband
from smac.intensification.successive_halving import SuccessiveHalving
from smac.runhistory.runhistory import RunHistory
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

        self.rh = RunHistory()
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

    def test_update_stage(self):
        """
            test initialization of all parameters and tracking variables
        """
        intensifier = Hyperband(
            tae_runner=None, stats=self.stats, traj_logger=None,
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[1], initial_budget=0.1, max_budget=1, eta=2)
        intensifier._update_stage()

        self.assertEqual(intensifier.s, 3)
        self.assertEqual(intensifier.s_max, 3)
        self.assertEqual(intensifier.hb_iters, 0)
        self.assertIsInstance(intensifier.sh_intensifier, SuccessiveHalving)
        self.assertEqual(intensifier.sh_intensifier.initial_budget, 0.125)
        self.assertEqual(intensifier.sh_intensifier.n_configs_in_stage, [8.0, 4.0, 2.0, 1.0])

        # next HB stage
        intensifier._update_stage()

        self.assertEqual(intensifier.s, 2)
        self.assertEqual(intensifier.hb_iters, 0)
        self.assertEqual(intensifier.sh_intensifier.initial_budget, 0.25)
        self.assertEqual(intensifier.sh_intensifier.n_configs_in_stage, [4.0, 2.0, 1.0])

        intensifier._update_stage()  # s = 1
        intensifier._update_stage()  # s = 0
        # HB iteration completed
        intensifier._update_stage()

        self.assertEqual(intensifier.s, intensifier.s_max)
        self.assertEqual(intensifier.hb_iters, 1)
        self.assertEqual(intensifier.sh_intensifier.initial_budget, 0.125)
        self.assertEqual(intensifier.sh_intensifier.n_configs_in_stage, [8.0, 4.0, 2.0, 1.0])

    def test_eval_challenger(self):
        """
            since hyperband uses eval_challenger and get_next_challenger of the internal successive halving,
            we don't test these method extensively
        """

        def target(x):
            return 0.1

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = Hyperband(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345), deterministic=True, run_obj_time=False,
            instances=[None], initial_budget=0.5, max_budget=1, eta=2)

        self.assertFalse(hasattr(intensifier, 's'))

        # Testing get_next_challenger - get next configuration
        config, _ = intensifier.get_next_challenger(challengers=[self.config2, self.config3],
                                                    chooser=None, run_history=self.rh)
        self.assertEqual(intensifier.s, intensifier.s_max)
        self.assertEqual(config, self.config2)

        # update to the last SH iteration of the given HB stage
        self.assertEqual(intensifier.s, 1)
        self.assertEqual(intensifier.s_max, 1)

        self.rh.add(config=self.config1, cost=1, time=1, status=StatusType.SUCCESS,
                    seed=0, budget=1)
        self.rh.add(config=self.config2, cost=2, time=2, status=StatusType.SUCCESS,
                    seed=0, budget=0.5)
        self.rh.add(config=self.config3, cost=3, time=2, status=StatusType.SUCCESS,
                    seed=0, budget=0.5)
        intensifier.sh_intensifier.success_challengers = {self.config2, self.config3}
        intensifier.sh_intensifier._update_stage(self.rh)

        # evaluation should change the incumbent to config2
        inc, inc_value = intensifier.eval_challenger(challenger=self.config2,
                                                     incumbent=self.config1,
                                                     run_history=self.rh)

        self.assertEqual(inc, self.config2)
        self.assertEqual(intensifier.s, 0)
        self.assertEqual(inc_value, 0.1)
        self.assertEqual(self.stats.ta_runs, 1)
        self.assertEqual(list(self.rh.data.keys())[-1][0], self.rh.config_ids[self.config2])
        self.assertEqual(self.stats.inc_changed, 1)
