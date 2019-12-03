import shutil
import unittest
from unittest import mock

import numpy as np
from nose.plugins.attrib import attr

from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
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

from test import requires_extra


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
        patch.side_effect = FirstRunCrashedException()
        scen = Scenario({'cs': test_helpers.get_branin_config_space(),
                         'run_obj': 'quality', 'output_dir': 'data-test_smbo-abort',
                         'abort_on_first_run_crash': 1})
        self.output_dirs.append(scen.output_dir)
        smbo = SMAC4AC(scen, tae_runner=target, rng=1).solver

        with self.assertRaises(FirstRunCrashedException):
            smbo.start()
            smbo.run()

    @attr('slow')
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
        self.assertAlmostEqual(3.0, smbo._get_timebound_for_intensification(7.0))
        smbo = get_smbo(0.5)
        self.assertAlmostEqual(0.03, smbo._get_timebound_for_intensification(0.03))
        smbo = get_smbo(0.7)
        self.assertAlmostEqual(1.4, smbo._get_timebound_for_intensification(0.6))
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

    def test_rf_comp_builder(self):
        seed = 42
        smbo = SMAC4AC(self.scenario, rng=seed).solver
        conf = {"model": "RF", "acq_func": "EI"}
        acqf, model = smbo._component_builder(conf)

        self.assertTrue(isinstance(acqf, EI))
        self.assertTrue(isinstance(model, RandomForestWithInstances))

    @requires_extra('gp')
    def test_gp_comp_builder(self):
        seed = 42
        smbo = SMAC4AC(self.scenario, rng=seed).solver
        conf = {"model": "GP", "acq_func": "EI"}
        acqf, model = smbo._component_builder(conf)

        self.assertTrue(isinstance(acqf, EI))
        self.assertTrue(isinstance(model, GaussianProcessMCMC))

    def test_smbo_cs(self):
        seed = 42
        smbo = SMAC4AC(self.scenario, rng=seed).solver
        _ = smbo._get_acm_cs()

    def test_cs_comp_builder(self):
        seed = 42
        smbo = SMAC4AC(self.scenario, rng=seed).solver
        cs = smbo._get_acm_cs()
        conf = cs.sample_configuration()

        acqf, model = smbo._component_builder(conf)


if __name__ == "__main__":
    unittest.main()
