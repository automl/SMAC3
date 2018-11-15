import json
import os
import sys
import unittest
import shutil
from nose.plugins.attrib import attr

from unittest import mock

from smac.smac_cli import SMACCLI
from smac.optimizer import ei_optimization
from ConfigSpace.util import get_one_exchange_neighbourhood


class TestDeterministicSMAC(unittest.TestCase):

    def setUp(self):
        base_directory = os.path.split(__file__)[0]
        base_directory = os.path.abspath(
            os.path.join(base_directory, '..', '..'))
        self.current_dir = os.getcwd()
        os.chdir(base_directory)

        self.output_dir_1 = "test/test_files/out_test_deterministic_1"
        self.output_dir_2 = "test/test_files/out_test_deterministic_2"
        self.output_dir_3 = "test/test_files/out_test_deterministic_3"
        self.scenario_file = "test/test_files/test_deterministic_scenario.txt"
        self.output_dirs = [self.output_dir_1, self.output_dir_2, self.output_dir_3]

        self.maxDiff = None

    def tearDown(self):
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)
        os.chdir(self.current_dir)

    @attr('slow')
    @unittest.mock.patch("smac.optimizer.ei_optimization.get_one_exchange_neighbourhood")
    def test_deterministic(self, patch):
        """
        Testing deterministic behaviour.
        """

        # Make SMAC a bit faster
        patch.side_effect = lambda configuration, seed: get_one_exchange_neighbourhood(
            configuration=configuration,
            stdev=0.05,
            num_neighbors=2,
            seed=seed,
        )

        testargs = ["scripts/smac",
                    "--scenario", self.scenario_file,
                    "--verbose_level", "DEBUG",
                    "--seed", "1",
                    "--random_configuration_chooser", "test/test_cli/random_configuration_chooser_impl.py",
                    "--output_dir", self.output_dir_1]
        SMACCLI().main_cli(testargs[1:])
        testargs = ["scripts/smac",
                    "--scenario", self.scenario_file,
                    "--verbose_level", "DEBUG",
                    "--seed", "1",
                    "--random_configuration_chooser", "test/test_cli/random_configuration_chooser_impl.py",
                    "--output_dir", self.output_dir_2]
        SMACCLI().main_cli(testargs[1:])
        testargs = ["scripts/smac",
                    "--scenario", self.scenario_file,
                    "--verbose_level", "DEBUG",
                    "--seed", "2",
                    "--random_configuration_chooser", "test/test_cli/random_configuration_chooser_impl.py",
                    "--output_dir", self.output_dir_3]
        SMACCLI().main_cli(testargs[1:])
        # compare trajectories in output_dir_{1,2,3}
        h1 = json.load(open(self.output_dir_1 + '/run_1/runhistory.json'))
        h2 = json.load(open(self.output_dir_2 + '/run_1/runhistory.json'))
        h3 = json.load(open(self.output_dir_3 + '/run_2/runhistory.json'))
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

    def test_modes(self):
        """
        Test if different modes are accepted
        """
        testargs = ["scripts/smac",
                    "--scenario", self.scenario_file,
                    "--verbose_level", "DEBUG",
                    "--seed", "2",
                    "--random_configuration_chooser", "test/test_cli/random_configuration_chooser_impl.py",
                    "--output_dir", self.output_dir_3,
                    "--mode", 'SMAC']
        cli = SMACCLI()
        with mock.patch("smac.smac_cli.SMAC") as MSMAC:
            MSMAC.return_value.optimize.return_value = True
            cli.main_cli(testargs[1:])
            MSMAC.assert_called_once()

        testargs[-1] = 'BOGP'
        cli = SMACCLI()
        with mock.patch("smac.smac_cli.BOGP") as MSMAC:
            MSMAC.return_value.optimize.return_value = True
            cli.main_cli(testargs[1:])
            MSMAC.assert_called_once()

        testargs[-1] = 'BORF'
        cli = SMACCLI()
        with mock.patch("smac.smac_cli.BORF") as MSMAC:
            MSMAC.return_value.optimize.return_value = True
            cli.main_cli(testargs[1:])
            MSMAC.assert_called_once()

        testargs[-1] = 'Hydra'
        cli = SMACCLI()
        with mock.patch("smac.smac_cli.Hydra") as MSMAC:
            MSMAC.return_value.optimize.return_value = True
            cli.main_cli(testargs[1:])
            MSMAC.assert_called_once()

        testargs[-1] = 'PSMAC'
        cli = SMACCLI()
        with mock.patch("smac.smac_cli.PSMAC") as MSMAC:
            MSMAC.return_value.optimize.return_value = True
            cli.main_cli(testargs[1:])
            MSMAC.assert_called_once()
