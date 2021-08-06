import unittest

import logging
import numpy as np

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.intensification.abstract_racer import AbstractRacer
from smac.tae import StatusType
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='a',
                                                       lower=0,
                                                       upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='b',
                                                       lower=0,
                                                       upper=100))
    return cs


class TestAbstractRacer(unittest.TestCase):

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

    def test_compare_configs_no_joint_set(self):
        intensifier = AbstractRacer(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=None, instances=[1])

        for i in range(2):
            self.rh.add(config=self.config1, cost=2, time=2,
                        status=StatusType.SUCCESS, instance_id=1,
                        seed=i, additional_info=None)

        for i in range(2, 5):
            self.rh.add(config=self.config2, cost=1, time=1,
                        status=StatusType.SUCCESS, instance_id=1,
                        seed=i, additional_info=None)

        # The sets for the incumbent are completely disjoint.
        conf = intensifier._compare_configs(incumbent=self.config1,
                                            challenger=self.config2,
                                            run_history=self.rh)
        self.assertIsNone(conf)

        # The incumbent has still one instance-seed pair left on which the
        # challenger was not run yet.
        self.rh.add(config=self.config2, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=1, additional_info=None)
        conf = intensifier._compare_configs(incumbent=self.config1,
                                            challenger=self.config2,
                                            run_history=self.rh)
        self.assertIsNone(conf)

    def test_compare_configs_chall(self):
        """
            challenger is better
        """
        intensifier = AbstractRacer(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=None,
            instances=[1])

        self.rh.add(config=self.config1, cost=1, time=2,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config2, cost=0, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        conf = intensifier._compare_configs(incumbent=self.config1,
                                            challenger=self.config2,
                                            run_history=self.rh)

        # challenger has enough runs and is better
        self.assertEqual(conf, self.config2, "conf: %s" % (conf))

    def test_compare_configs_inc(self):
        """
            incumbent is better
        """
        intensifier = AbstractRacer(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=None,
            instances=[1])

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        conf = intensifier._compare_configs(incumbent=self.config1,
                                            challenger=self.config2,
                                            run_history=self.rh)

        # challenger worse than inc
        self.assertEqual(conf, self.config1, "conf: %s" % (conf))

    def test_compare_configs_unknow(self):
        """
            challenger is better but has less runs;
            -> no decision (None)
        """
        intensifier = AbstractRacer(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=None,
            instances=[1])

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config1, cost=1, time=2,
                    status=StatusType.SUCCESS, instance_id=2,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=2,
                    seed=None,
                    additional_info=None)

        conf = intensifier._compare_configs(incumbent=self.config1,
                                            challenger=self.config2,
                                            run_history=self.rh)

        # challenger worse than inc
        self.assertIsNone(conf, "conf: %s" % (conf))

    def test_adaptive_capping(self):
        """
            test _adapt_cutoff()
        """
        intensifier = AbstractRacer(
            stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=list(range(5)),
            deterministic=False)

        for i in range(5):
            self.rh.add(config=self.config1, cost=i + 1, time=i + 1,
                        status=StatusType.SUCCESS, instance_id=i,
                        seed=i,
                        additional_info=None)
        for i in range(3):
            self.rh.add(config=self.config2, cost=i + 1, time=i + 1,
                        status=StatusType.SUCCESS, instance_id=i,
                        seed=i,
                        additional_info=None)

        inst_seed_pairs = self.rh.get_runs_for_config(self.config1, only_max_observed_budget=True)
        # cost used by incumbent for going over all runs in inst_seed_pairs
        inc_sum_cost = self.rh.sum_cost(config=self.config1, instance_seed_budget_keys=inst_seed_pairs)

        cutoff = intensifier._adapt_cutoff(challenger=self.config2, run_history=self.rh, inc_sum_cost=inc_sum_cost)
        # 15*1.2 - 6
        self.assertEqual(cutoff, 12)

        intensifier.cutoff = 5

        cutoff = intensifier._adapt_cutoff(challenger=self.config2, run_history=self.rh, inc_sum_cost=inc_sum_cost)
        # scenario cutoff
        self.assertEqual(cutoff, 5)
