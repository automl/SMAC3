import os
import sys
import unittest
from nose.plugins.attrib import attr
import shutil

import numpy as np

from unittest import mock

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace
from smac.smac_cli import SMACCLI
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.optimizer.smbo import SMBO
from smac.stats.stats import Stats


class TestSMACCLI(unittest.TestCase):

    def setUp(self):
        base_directory = os.path.split(__file__)[0]
        base_directory = os.path.abspath(
            os.path.join(base_directory, '..', '..'))
        self.current_dir = os.getcwd()
        os.chdir(base_directory)

        output_one_dir = "test/test_files/test_restore_state"  # From scenario_one.txt
        self.output_one = output_one_dir + "/run_1"
        output_two_dir = "test/test_files/test_restored_state"  # From scenario_two.txt
        self.output_two = output_two_dir + "/run_1"
        self.smaccli = SMACCLI()
        self.scenario_one = "test/test_files/restore_scenario_one.txt"
        self.scenario_two = "test/test_files/restore_scenario_two.txt"
        self.output_dirs = [output_one_dir, output_two_dir]

    def tearDown(self):
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)
                #pass
        os.chdir(self.current_dir)

    @attr('slow')
    def test_run_and_restore(self):
        """
        Testing basic restore functionality.
        """
        # Run for 5 algo-calls
        testargs = ["python", "scripts/smac", "--scenario_file",
                    self.scenario_one, "--verbose", "DEBUG"]
        self.smaccli.main_cli(testargs[2:])
        # Increase limit and run for 10 (so 5 more) by using restore_state
        testargs = ["python", "scripts/smac", "--restore_state",
                    self.output_one, "--scenario_file",
                    self.scenario_two, "--verbose", "DEBUG"]
        self.smaccli.main_cli(testargs[2:])

    def test_missing_dir(self):
        """
        Testing error if dir is missing.
        """
        testargs = ["python", "scripts/smac", "--restore_state",
                    "nonsense_test_dir", "--scenario_file",
                    self.scenario_two, "--verbose", "DEBUG"]
        self.assertRaises(FileNotFoundError, lambda: self.smaccli.main_cli(testargs[2:]))

    def test_illegal_input(self):
        """
        Testing illegal input in smbo
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformFloatHyperparameter('test', 1, 10, 5))
        scen = Scenario({'run_obj': 'quality', 'cs': cs})
        stats = Stats(scen)
        # Recorded runs but no incumbent.
        stats.ta_runs = 10
        smac = SMAC(scen, stats=stats, rng=np.random.RandomState(42))
        self.output_dirs.append(scen.output_dir)
        self.assertRaises(ValueError, smac.optimize)
        # Incumbent but no recoreded runs.
        incumbent = cs.get_default_configuration()
        smac = SMAC(scen, restore_incumbent=incumbent,
                    rng=np.random.RandomState(42))
        self.assertRaises(ValueError, smac.optimize)

    @attr('slow')
    def test_same_dir(self):
        """
        Testing possible error using same dir for restore
        """
        # Run for 5 algo-calls
        testargs = ["python", "scripts/smac", "--scenario",
                    self.scenario_one, "--verbose", "DEBUG"]
        self.smaccli.main_cli(testargs[2:])
        # Increase limit and run for 10 (so 5 more) by using restore_state
        testargs = ["python", "scripts/smac", "--restore_state",
                    self.output_one, "--scenario",
                    self.scenario_two, "--verbose", "DEBUG"]
        # TODO: fix
        try:
            self.smaccli.main_cli(testargs[2:])
        except FileNotFoundError:
            pass
        self.assertTrue(os.path.exists(self.output_one))
        self.assertFalse(os.path.exists(self.output_one + '.OLD'))
        self.assertTrue(os.path.exists(self.output_two))
        self.assertFalse(os.path.exists(self.output_two + '.OLD'))


