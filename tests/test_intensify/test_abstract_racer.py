import logging
import unittest

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.intensification import AbstractIntensifier
from smac.runhistory.runhistory import RunHistory
from smac.runner.runner import StatusType
from smac.scenario import Scenario
from smac.utils.stats import Stats

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name="a", lower=0, upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name="b", lower=0, upper=100))
    return cs


class TestAbstractRacer(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)

        self.rh = RunHistory()
        self.cs = get_config_space()
        self.config1 = Configuration(self.cs, values={"a": 0, "b": 100})
        self.config2 = Configuration(self.cs, values={"a": 100, "b": 0})
        self.config3 = Configuration(self.cs, values={"a": 100, "b": 100})

        scenario = Scenario(self.cs, algorithm_walltime_limit=2, output_directory="smac3_output_test")
        self.stats = Stats(scenario=scenario)
        self.intensifier = AbstractIntensifier(scenario=scenario)
        self.intensifier._set_stats(self.stats)

        self.stats.start_timing()

    def test_compare_configs_no_joint_set(self):
        for i in range(2):
            self.rh.add(
                config=self.config1,
                cost=2,
                time=2,
                status=StatusType.SUCCESS,
                instance_id=1,
                seed=i,
                additional_info=None,
            )

        for i in range(2, 5):
            self.rh.add(
                config=self.config2,
                cost=1,
                time=1,
                status=StatusType.SUCCESS,
                instance_id=1,
                seed=i,
                additional_info=None,
            )

        # The sets for the incumbent are completely disjoint.
        conf = self.intensifier._compare_configs(incumbent=self.config1, challenger=self.config2, runhistory=self.rh)
        self.assertIsNone(conf)

        # The incumbent has still one instance-seed pair left on which the
        # challenger was not run yet.
        self.rh.add(
            config=self.config2,
            cost=1,
            time=1,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            additional_info=None,
        )
        conf = self.intensifier._compare_configs(incumbent=self.config1, challenger=self.config2, runhistory=self.rh)
        self.assertIsNone(conf)

    def test_compare_configs_chall(self):
        """
        Challenger is better.
        """
        self.rh.add(
            config=self.config1,
            cost=1,
            time=2,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=None,
            additional_info=None,
        )

        self.rh.add(
            config=self.config2,
            cost=0,
            time=1,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=None,
            additional_info=None,
        )

        conf = self.intensifier._compare_configs(incumbent=self.config1, challenger=self.config2, runhistory=self.rh)

        # challenger has enough runs and is better
        self.assertEqual(conf, self.config2, "conf: %s" % (conf))

    def test_compare_configs_inc(self):
        """
        Incumbent is better
        """

        self.rh.add(
            config=self.config1,
            cost=1,
            time=1,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=None,
            additional_info=None,
        )

        self.rh.add(
            config=self.config2,
            cost=2,
            time=2,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=None,
            additional_info=None,
        )

        conf = self.intensifier._compare_configs(incumbent=self.config1, challenger=self.config2, runhistory=self.rh)

        # challenger worse than inc
        self.assertEqual(conf, self.config1, "conf: %s" % (conf))

    def test_compare_configs_unknow(self):
        """
        Challenger is better but has less runs;
        -> no decision (None)
        """

        self.rh.add(
            config=self.config1,
            cost=1,
            time=1,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=None,
            additional_info=None,
        )

        self.rh.add(
            config=self.config1,
            cost=1,
            time=2,
            status=StatusType.SUCCESS,
            instance_id=2,
            seed=None,
            additional_info=None,
        )

        self.rh.add(
            config=self.config1,
            cost=1,
            time=1,
            status=StatusType.SUCCESS,
            instance_id=2,
            seed=None,
            additional_info=None,
        )

        conf = self.intensifier._compare_configs(incumbent=self.config1, challenger=self.config2, runhistory=self.rh)

        # challenger worse than inc
        self.assertIsNone(conf, "conf: %s" % (conf))
