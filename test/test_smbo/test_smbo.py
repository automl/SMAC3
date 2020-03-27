import shutil
import unittest
from unittest import mock

import numpy as np

from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.intensification import Intensifier
from smac.optimizer.acquisition import EI, LogEI
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import FirstRunCrashedException
from smac.utils import test_helpers
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator


class ConfigurationMock(object):
    def __init__(self, value=None):
        self.value = value

    def get_array(self):
        return [self.value]


class TestSMBO(unittest.TestCase):

    def setUp(self):
        self.scenario = Scenario({'cs': test_helpers.get_branin_config_space(),
                                  'run_obj': 'quality',
                                  'output_dir': 'data-test_smbo'})
        self.output_dirs = []
        self.output_dirs.append(self.scenario.output_dir)

    def tearDown(self):
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)

    def branin(self, x):
        y = (x[:, 1] - (5.1 / (4 * np.pi ** 2)) * x[:, 0] ** 2 + 5 * x[:, 0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:, 0]) + 10

        return y[:, np.newaxis]

    def test_init_only_scenario_runtime(self):
        self.scenario.run_obj = 'runtime'
        self.scenario.cutoff = 300
        smbo = SMAC4AC(self.scenario).solver
        self.assertIsInstance(smbo.epm_chooser.model, RandomForestWithInstances)
        self.assertIsInstance(smbo.epm_chooser.rh2EPM, RunHistory2EPM4LogCost)
        self.assertIsInstance(smbo.epm_chooser.acquisition_func, LogEI)

    def test_init_only_scenario_quality(self):
        smbo = SMAC4AC(self.scenario).solver
        self.assertIsInstance(smbo.epm_chooser.model, RandomForestWithInstances)
        self.assertIsInstance(smbo.epm_chooser.rh2EPM, RunHistory2EPM4Cost)
        self.assertIsInstance(smbo.epm_chooser.acquisition_func, EI)

    def test_rng(self):
        smbo = SMAC4AC(self.scenario, rng=None).solver
        self.assertIsInstance(smbo.rng, np.random.RandomState)
        self.assertIsInstance(smbo.num_run, int)
        smbo = SMAC4AC(self.scenario, rng=1).solver
        rng = np.random.RandomState(1)
        self.assertEqual(smbo.num_run, 1)
        self.assertIsInstance(smbo.rng, np.random.RandomState)
        smbo = SMAC4AC(self.scenario, rng=rng).solver
        self.assertIsInstance(smbo.num_run, int)
        self.assertIs(smbo.rng, rng)
        # ML: I don't understand the following line and it throws an error
        self.assertRaisesRegex(
            TypeError,
            "Argument rng accepts only arguments of type None, int or np.random.RandomState, you provided "
            "<class 'str'>.",
            SMAC4AC,
            self.scenario,
            rng='BLA',
        )

    @mock.patch.object(Intensifier, 'eval_challenger')
    def test_abort_on_initial_design(self, patch):
        def target(x):
            return 5

        # should raise an error if abort_on_first_run_crash is True
        patch.side_effect = FirstRunCrashedException()
        scen = Scenario({'cs': test_helpers.get_branin_config_space(),
                         'run_obj': 'quality', 'output_dir': 'data-test_smbo-abort',
                         'abort_on_first_run_crash': True})
        self.output_dirs.append(scen.output_dir)
        smbo = SMAC4AC(scen, tae_runner=target, rng=1).solver
        self.assertRaises(FirstRunCrashedException, smbo.run)

        # should not raise an error if abort_on_first_run_crash is False
        patch.side_effect = FirstRunCrashedException()
        scen = Scenario({'cs': test_helpers.get_branin_config_space(),
                         'run_obj': 'quality', 'output_dir': 'data-test_smbo-abort',
                         'abort_on_first_run_crash': False, 'wallclock-limit': 1})
        self.output_dirs.append(scen.output_dir)
        smbo = SMAC4AC(scen, tae_runner=target, rng=1).solver

        try:
            smbo.start()
            smbo.run()
        except FirstRunCrashedException:
            self.fail('Raises FirstRunCrashedException unexpectedly!')

    def test_intensification_percentage(self):
        def target(x):
            return 5

        def get_smbo(intensification_perc):
            """ Return SMBO with intensification_percentage. """
            scen = Scenario({'cs': test_helpers.get_branin_config_space(),
                             'run_obj': 'quality', 'output_dir': 'data-test_smbo-intensification',
                             'intensification_percentage': intensification_perc})
            self.output_dirs.append(scen.output_dir)
            return SMAC4AC(scen, tae_runner=target, rng=1).solver
        # Test for valid values
        smbo = get_smbo(0.3)
        self.assertAlmostEqual(3.0, smbo._get_timebound_for_intensification(7.0, update=False))
        smbo = get_smbo(0.5)
        self.assertAlmostEqual(0.03, smbo._get_timebound_for_intensification(0.03, update=False))
        smbo = get_smbo(0.7)
        self.assertAlmostEqual(1.4, smbo._get_timebound_for_intensification(0.6, update=False))
        # Test for invalid <= 0
        smbo = get_smbo(0)
        self.assertRaises(ValueError, smbo.run)
        smbo = get_smbo(-0.2)
        self.assertRaises(ValueError, smbo.run)
        # Test for invalid >= 1
        smbo = get_smbo(1)
        self.assertRaises(ValueError, smbo.run)
        smbo = get_smbo(1.2)
        self.assertRaises(ValueError, smbo.run)

    def test_update_intensification_percentage(self):
        """
        This test checks the insification time bound is updated in subsequent iterations as long as
        num_runs of the intensifier is not reset to zero.
        """

        def target(x):
            return 5

        scen = Scenario({'cs': test_helpers.get_branin_config_space(),
                         'run_obj': 'quality', 'output_dir': 'data-test_smbo-intensification'})
        self.output_dirs.append(scen.output_dir)
        solver = SMAC4AC(scen, tae_runner=target, rng=1).solver

        solver.stats.is_budget_exhausted = unittest.mock.Mock()
        solver.stats.is_budget_exhausted.side_effect = tuple(([False] * 10) + [True] * 8)

        solver._get_timebound_for_intensification = unittest.mock.Mock(wraps=solver._get_timebound_for_intensification)

        class SideEffect:
            def __init__(self, intensifier, get_next_challenger):
                self.intensifier = intensifier
                self.get_next_challenger = get_next_challenger
                self.counter = 0

            def __call__(self, *args, **kwargs):
                self.counter += 1
                if self.counter % 4 == 0:
                    self.intensifier.num_run = 0
                return self.get_next_challenger(*args, **kwargs)

        solver.intensifier.get_next_challenger = unittest.mock.Mock(
            side_effect=SideEffect(solver.intensifier, solver.intensifier.get_next_challenger))

        solver.run()

        get_timebound_mock = solver._get_timebound_for_intensification
        self.assertEqual(get_timebound_mock.call_count, 6)
        self.assertFalse(get_timebound_mock.call_args_list[0][1]['update'])
        self.assertFalse(get_timebound_mock.call_args_list[1][1]['update'])
        self.assertTrue(get_timebound_mock.call_args_list[2][1]['update'])
        self.assertFalse(get_timebound_mock.call_args_list[3][1]['update'])
        self.assertTrue(get_timebound_mock.call_args_list[4][1]['update'])
        self.assertTrue(get_timebound_mock.call_args_list[5][1]['update'])

        self.assertGreater(get_timebound_mock.call_args_list[2][0][0], get_timebound_mock.call_args_list[1][0][0])
        self.assertLess(get_timebound_mock.call_args_list[3][0][0], get_timebound_mock.call_args_list[2][0][0])
        self.assertGreater(get_timebound_mock.call_args_list[4][0][0], get_timebound_mock.call_args_list[3][0][0])
        self.assertGreater(get_timebound_mock.call_args_list[5][0][0], get_timebound_mock.call_args_list[4][0][0])

    def test_validation(self):
        with mock.patch.object(TrajLogger, "read_traj_aclib_format",
                               return_value=None):
            self.scenario.output_dir = "test"
            smac = SMAC4AC(self.scenario)
            self.output_dirs.append(smac.output_dir)
            smbo = smac.solver
            with mock.patch.object(Validator, "validate", return_value=None) as validation_mock:
                smbo.validate(config_mode='inc', instance_mode='train+test',
                              repetitions=1, use_epm=False, n_jobs=-1, backend='threading')
                self.assertTrue(validation_mock.called)
            with mock.patch.object(Validator, "validate_epm", return_value=None) as epm_validation_mock:
                smbo.validate(config_mode='inc', instance_mode='train+test',
                              repetitions=1, use_epm=True, n_jobs=-1, backend='threading')
                self.assertTrue(epm_validation_mock.called)

    def test_no_initial_design(self):
        self.scenario.output_dir = "test"
        smac = SMAC4AC(self.scenario)
        self.output_dirs.append(smac.output_dir)
        smbo = smac.solver
        # SMBO should have the default configuration as the 1st config if no initial design is given
        smbo.start()
        self.assertEqual(smbo.initial_design_configs[0], smbo.scenario.cs.get_default_configuration())
