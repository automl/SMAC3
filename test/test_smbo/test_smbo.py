from contextlib import suppress
import unittest
from unittest import mock
import os
import shutil
from nose.plugins.attrib import attr

import numpy as np
from ConfigSpace import Configuration

from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.uncorrelated_mo_rf_with_instances import \
    UncorrelatedMultiObjectiveRandomForestWithInstances
from smac.facade.smac_facade import SMAC
from smac.initial_design.single_config_initial_design import SingleConfigInitialDesign
from smac.optimizer.acquisition import EI, EIPS, LogEI
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, \
    RunHistory2EPM4LogCost, RunHistory2EPM4EIPS
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import FirstRunCrashedException
from smac.utils import test_helpers
from smac.utils.util_funcs import get_types
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
        smbo = SMAC(self.scenario).solver
        self.assertIsInstance(smbo.model, RandomForestWithInstances)
        self.assertIsInstance(smbo.rh2EPM, RunHistory2EPM4LogCost)
        self.assertIsInstance(smbo.acquisition_func, LogEI)

    def test_init_only_scenario_quality(self):
        smbo = SMAC(self.scenario).solver
        self.assertIsInstance(smbo.model, RandomForestWithInstances)
        self.assertIsInstance(smbo.rh2EPM, RunHistory2EPM4Cost)
        self.assertIsInstance(smbo.acquisition_func, EI)

    def test_init_EIPS_as_arguments(self):
        for objective in ['runtime', 'quality']:
            self.scenario.run_obj = objective
            types, bounds = get_types(self.scenario.cs, None)
            umrfwi = UncorrelatedMultiObjectiveRandomForestWithInstances(
                ['cost', 'runtime'], types, bounds)
            eips = EIPS(umrfwi)
            rh2EPM = RunHistory2EPM4EIPS(self.scenario, 2)
            smbo = SMAC(self.scenario, model=umrfwi, acquisition_function=eips,
                        runhistory2epm=rh2EPM).solver
            self.assertIs(umrfwi, smbo.model)
            self.assertIs(eips, smbo.acquisition_func)
            self.assertIs(rh2EPM, smbo.rh2EPM)

    def test_rng(self):
        smbo = SMAC(self.scenario, rng=None).solver
        self.assertIsInstance(smbo.rng, np.random.RandomState)
        self.assertIsInstance(smbo.num_run, int)
        smbo = SMAC(self.scenario, rng=1).solver
        rng = np.random.RandomState(1)
        self.assertEqual(smbo.num_run, 1)
        self.assertIsInstance(smbo.rng, np.random.RandomState)
        smbo = SMAC(self.scenario, rng=rng).solver
        self.assertIsInstance(smbo.num_run, int)
        self.assertIs(smbo.rng, rng)
        # ML: I don't understand the following line and it throws an error
        self.assertRaisesRegex(
            TypeError,
            "Argument rng accepts only arguments of type None, int or np.random.RandomState, you provided "
            "<class 'str'>.",
            SMAC,
            self.scenario,
            rng='BLA',
        )

    def test_choose_next(self):
        seed = 42
        smbo = SMAC(self.scenario, rng=seed).solver
        smbo.runhistory = RunHistory(aggregate_func=average_cost)
        X = self.scenario.cs.sample_configuration().get_array()[None, :]
        smbo.incumbent = self.scenario.cs.sample_configuration()
        smbo.runhistory.add(smbo.incumbent, 10, 10, 1)

        Y = self.branin(X)
        x = next(smbo.choose_next(X, Y)).get_array()
        assert x.shape == (2,)

    def test_choose_next_w_empty_rh(self):
        seed = 42
        smbo = SMAC(self.scenario, rng=seed).solver
        smbo.runhistory = RunHistory(aggregate_func=average_cost)
        X = self.scenario.cs.sample_configuration().get_array()[None, :]

        Y = self.branin(X)
        self.assertRaisesRegex(
            ValueError,
            'Runhistory is empty and the cost value of the incumbent is '
            'unknown.',
            smbo.choose_next,
            **{"X":X, "Y":Y}
        )

        x = next(smbo.choose_next(X, Y, incumbent_value=0.0)).get_array()
        assert x.shape == (2,)

    def test_choose_next_empty_X(self):
        smbo = SMAC(self.scenario, rng=1).solver
        smbo.acquisition_func._compute = mock.Mock(
            spec=RandomForestWithInstances
        )
        smbo._random_search.maximize = mock.Mock(
            spec=smbo._random_search.maximize
        )
        smbo._random_search.maximize.return_value = [0, 1, 2]

        X = np.zeros((0, 2))
        Y = np.zeros((0, 1))

        x = smbo.choose_next(X, Y)
        self.assertEqual(x, [0, 1, 2])
        self.assertEqual(smbo._random_search.maximize.call_count, 1)
        self.assertEqual(smbo.acquisition_func._compute.call_count, 0)

    def test_choose_next_empty_X_2(self):
        smbo = SMAC(self.scenario, rng=1).solver

        X = np.zeros((0, 2))
        Y = np.zeros((0, 1))

        challengers = smbo.choose_next(X, Y)
        x = [c for c in challengers]
        self.assertEqual(len(x), 1)
        self.assertIsInstance(x[0], Configuration)

    def test_choose_next_2(self):
        # Test with a single configuration in the runhistory!
        def side_effect(X):
            return np.mean(X, axis=1).reshape((-1, 1))

        def side_effect_predict(X):
            m, v = np.ones((X.shape[0], 1)), None
            return m, v

        smbo = SMAC(self.scenario, rng=1).solver
        smbo.incumbent = self.scenario.cs.sample_configuration()
        smbo.runhistory = RunHistory(aggregate_func=average_cost)
        smbo.runhistory.add(smbo.incumbent, 10, 10, 1)
        smbo.model = mock.Mock(spec=RandomForestWithInstances)
        smbo.model.predict_marginalized_over_instances.side_effect = side_effect_predict
        smbo.acquisition_func._compute = mock.Mock(spec=RandomForestWithInstances)
        smbo.acquisition_func._compute.side_effect = side_effect

        X = smbo.rng.rand(10, 2)
        Y = smbo.rng.rand(10, 1)

        challengers = smbo.choose_next(X, Y)
        # Convert challenger list (a generator) to a real list
        challengers = [c for c in challengers]

        self.assertEqual(smbo.model.train.call_count, 1)

        # For each configuration it is randomly sampled whether to take it from the list of challengers or to sample it
        # completely at random. Therefore, it is not guaranteed to obtain twice the number of configurations selected
        # by EI.
        self.assertEqual(len(challengers), 9913)
        num_random_search_sorted = 0
        num_random_search = 0
        num_local_search = 0
        for c in challengers:
            self.assertIsInstance(c, Configuration)
            if 'Random Search (sorted)' == c.origin:
                num_random_search_sorted += 1
            elif 'Random Search' == c.origin:
                num_random_search += 1
            elif 'Local Search' == c.origin:
                num_local_search += 1
            else:
                raise ValueError(c.origin)

        self.assertEqual(num_local_search, 1)
        self.assertEqual(num_random_search_sorted, 4999)
        self.assertEqual(num_random_search, 4913)

    def test_choose_next_3(self):
        # Test with ten configurations in the runhistory
        def side_effect(X):
            return np.mean(X, axis=1).reshape((-1, 1))

        def side_effect_predict(X):
            m, v = np.ones((X.shape[0], 1)), None
            return m, v

        smbo = SMAC(self.scenario, rng=1).solver
        smbo.incumbent = self.scenario.cs.sample_configuration()
        previous_configs = [smbo.incumbent] + [self.scenario.cs.sample_configuration() for i in range(0, 20)]
        smbo.runhistory = RunHistory(aggregate_func=average_cost)
        for i in range(0, len(previous_configs)):
            smbo.runhistory.add(previous_configs[i], i, 10, 1)
        smbo.model = mock.Mock(spec=RandomForestWithInstances)
        smbo.model.predict_marginalized_over_instances.side_effect = side_effect_predict
        smbo.acquisition_func._compute = mock.Mock(spec=RandomForestWithInstances)
        smbo.acquisition_func._compute.side_effect = side_effect

        X = smbo.rng.rand(10, 2)
        Y = smbo.rng.rand(10, 1)

        challengers = smbo.choose_next(X, Y)
        # Convert challenger list (a generator) to a real list
        challengers = [c for c in challengers]

        self.assertEqual(smbo.model.train.call_count, 1)

        # For each configuration it is randomly sampled whether to take it from the list of challengers or to sample it
        # completely at random. Therefore, it is not guaranteed to obtain twice the number of configurations selected
        # by EI.
        self.assertEqual(len(challengers), 9913)
        num_random_search_sorted = 0
        num_random_search = 0
        num_local_search = 0
        for c in challengers:
            self.assertIsInstance(c, Configuration)
            if 'Random Search (sorted)' == c.origin:
                num_random_search_sorted += 1
            elif 'Random Search' == c.origin:
                num_random_search += 1
            elif 'Local Search' == c.origin:
                num_local_search += 1
            else:
                raise ValueError(c.origin)

        self.assertEqual(num_local_search, 10)
        self.assertEqual(num_random_search_sorted, 4990)
        self.assertEqual(num_random_search, 4913)

    @mock.patch.object(SingleConfigInitialDesign, 'run')
    def test_abort_on_initial_design(self, patch):
        def target(x):
            return 5
        patch.side_effect = FirstRunCrashedException()
        scen = Scenario({'cs': test_helpers.get_branin_config_space(),
                         'run_obj': 'quality', 'output_dir': 'data-test_smbo-abort',
                         'abort_on_first_run_crash': 1})
        self.output_dirs.append(scen.output_dir)
        smbo = SMAC(scen, tae_runner=target, rng=1).solver
        self.assertRaises(FirstRunCrashedException, smbo.run)

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
            return SMAC(scen, tae_runner=target, rng=1).solver
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
                return_value=None) as traj_mock:
            self.scenario.output_dir = "test"
            smac = SMAC(self.scenario)
            self.output_dirs.append(smac.output_dir)
            smbo = smac.solver
            with mock.patch.object(Validator, "validate",
                    return_value=None) as validation_mock:
                smbo.validate(config_mode='inc', instance_mode='train+test',
                              repetitions=1, use_epm=False, n_jobs=-1, backend='threading')
                self.assertTrue(validation_mock.called)
            with mock.patch.object(Validator, "validate_epm",
                    return_value=None) as epm_validation_mock:
                smbo.validate(config_mode='inc', instance_mode='train+test',
                              repetitions=1, use_epm=True, n_jobs=-1, backend='threading')
                self.assertTrue(epm_validation_mock.called)

    def test_no_initial_design(self):
        self.scenario.output_dir = "test"
        smac = SMAC(self.scenario)
        self.output_dirs.append(smac.output_dir)
        smbo = smac.solver
        with mock.patch.object(SingleConfigInitialDesign, "run", return_value=None) as initial_mock:
            smbo.start()
            self.assertEqual(smbo.incumbent, smbo.scenario.cs.get_default_configuration())


if __name__ == "__main__":
    unittest.main()
