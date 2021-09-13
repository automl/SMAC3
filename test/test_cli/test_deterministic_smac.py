import json
import os
import unittest
import shutil

from unittest import mock

from smac.smac_cli import SMACCLI
from ConfigSpace.util import get_one_exchange_neighbourhood

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


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

    def ignore_timestamps(self, rh):
        for i, (k, val) in enumerate(rh['data']):
            rh['data'][i][1] = [v for j, v in enumerate(val) if j not in [3, 4]]  # 3, 4 are start and end timestamps
        return rh

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
        self.assertEqual(self.ignore_timestamps(h1), self.ignore_timestamps(h2))
        # As h1 is changed inplace in the line above we need to reload it
        h1 = json.load(open(self.output_dir_1 + '/run_1/runhistory.json'))
        self.assertNotEqual(self.ignore_timestamps(h1), self.ignore_timestamps(h3))

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
                    "--mode", 'SMAC4AC']
        cli = SMACCLI()
        with mock.patch("smac.smac_cli.SMAC4AC") as MSMAC:
            MSMAC.return_value.optimize.return_value = True
            cli.main_cli(testargs[1:])
            MSMAC.assert_called_once_with(
                initial_configurations=None, restore_incumbent=None, run_id=2,
                runhistory=None, stats=None, scenario=mock.ANY, rng=mock.ANY)

        testargs[-1] = 'SMAC4BB'
        cli = SMACCLI()
        with mock.patch("smac.smac_cli.SMAC4BB") as MSMAC:
            MSMAC.return_value.optimize.return_value = True
            cli.main_cli(testargs[1:])
            MSMAC.assert_called_once_with(
                initial_configurations=None, restore_incumbent=None, run_id=2,
                runhistory=None, stats=None, scenario=mock.ANY, rng=mock.ANY)

        testargs[-1] = 'SMAC4HPO'
        cli = SMACCLI()
        with mock.patch("smac.smac_cli.SMAC4HPO") as MSMAC:
            MSMAC.return_value.optimize.return_value = True
            cli.main_cli(testargs[1:])
            MSMAC.assert_called_once_with(
                initial_configurations=None, restore_incumbent=None, run_id=2,
                runhistory=None, stats=None, scenario=mock.ANY, rng=mock.ANY)

        testargs[-1] = 'Hydra'
        cli = SMACCLI()
        with mock.patch("smac.smac_cli.Hydra") as MSMAC:
            MSMAC.return_value.optimize.return_value = True
            cli.main_cli(testargs[1:])
            MSMAC.assert_called_once_with(
                initial_configurations=None, restore_incumbent=None, run_id=2,
                incs_per_round=1, n_iterations=3,
                n_optimizers=1, random_configuration_chooser=mock.ANY,
                runhistory=None, stats=None, scenario=mock.ANY, rng=mock.ANY, val_set='train'
            )

        testargs[-1] = 'PSMAC'
        cli = SMACCLI()
        with mock.patch("smac.smac_cli.PSMAC") as MSMAC:
            MSMAC.return_value.optimize.return_value = True
            cli.main_cli(testargs[1:])
            MSMAC.assert_called_once_with(
                run_id=2, scenario=mock.ANY, rng=mock.ANY,
                n_incs=1, n_optimizers=1, shared_model=False, validate=False
            )
