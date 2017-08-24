import os
import sys
import unittest
import shutil
import glob
import re

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration

from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, \
    RunHistory2EPM4LogCost, RunHistory2EPM4EIPS
from smac.optimizer.smbo import SMBO
from smac.scenario.scenario import Scenario
from smac.optimizer.acquisition import EI, EIPS, LogEI
from smac.optimizer.local_search import LocalSearch
from smac.tae.execute_func import ExecuteTAFuncArray
from smac.tae.execute_ta_run import TAEAbortException, FirstRunCrashedException
from smac.stats.stats import Stats
from smac.utils import test_helpers
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.uncorrelated_mo_rf_with_instances import \
    UncorrelatedMultiObjectiveRandomForestWithInstances
from smac.utils.util_funcs import get_types
from smac.facade.smac_facade import SMAC
from smac.optimizer.objective import average_cost
from smac.initial_design.single_config_initial_design import SingleConfigInitialDesign
from smac.intensification.intensification import Intensifier

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock


class ConfigurationMock(object):
    def __init__(self, value=None):
        self.value = value

    def get_array(self):
        return [self.value]


class TestSMBO(unittest.TestCase):

    def setUp(self):
        self.scenario = Scenario({'cs': test_helpers.get_branin_config_space(),
                                  'run_obj': 'quality',
                                  'output_dir': ''})

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
        self.assertRaisesRegexp(TypeError,
                                "Unknown type <(class|type) 'str'> for argument "
                                'rng. Only accepts None, int or '
                                'np.random.RandomState',
                                SMAC, self.scenario, rng='BLA')

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
        self.assertRaises(ValueError, smbo.choose_next, **{"X":X, "Y":Y})

        x = next(smbo.choose_next(X, Y, incumbent_value=0.0)).get_array()        
        assert x.shape == (2,)

    def test_choose_next_2(self):
        def side_effect(X):
            return np.mean(X, axis=1).reshape((-1, 1))

        smbo = SMAC(self.scenario, rng=1).solver
        smbo.incumbent = self.scenario.cs.sample_configuration()
        smbo.runhistory = RunHistory(aggregate_func=average_cost)
        smbo.runhistory.add(smbo.incumbent, 10, 10, 1)
        smbo.model = mock.Mock(spec=RandomForestWithInstances)
        smbo.acquisition_func._compute = mock.Mock(spec=RandomForestWithInstances)
        smbo.acquisition_func._compute.side_effect = side_effect

        X = smbo.rng.rand(10, 2)
        Y = smbo.rng.rand(10, 1)

        challengers = smbo.choose_next(X, Y)
        x = [c for c in challengers]

        self.assertEqual(smbo.model.train.call_count, 1)

        self.assertEqual(len(x), 2002)
        num_random_search = 0
        num_local_search = 0
        for i in range(0, 2002, 2):
            # print(x[i].origin)
            self.assertIsInstance(x[i], Configuration)
            if 'Random Search (sorted)' in x[i].origin:
                num_random_search += 1
            elif 'Local Search' in x[i].origin:
                num_local_search += 1
        # number of local search configs has to be least 10
        # since x can have duplicates
        # which can be associated with the local search
        self.assertGreaterEqual(num_local_search, 1)
        for i in range(1, 2002, 2):
            self.assertIsInstance(x[i], Configuration)
            self.assertEqual(x[i].origin, 'Random Search')

    def test_choose_next_3(self):
        def side_effect(X):
            return np.mean(X, axis=1).reshape((-1, 1))

        smbo = SMAC(self.scenario, rng=1).solver
        smbo.incumbent = self.scenario.cs.sample_configuration()
        previous_configs = [smbo.incumbent] + [self.scenario.cs.sample_configuration() for i in range(0, 20)]
        smbo.runhistory = RunHistory(aggregate_func=average_cost)
        for i in range(0, len(previous_configs)):
            smbo.runhistory.add(previous_configs[i], i, 10, 1)
        smbo.model = mock.Mock(spec=RandomForestWithInstances)
        smbo.acquisition_func._compute = mock.Mock(spec=RandomForestWithInstances)
        smbo.acquisition_func._compute.side_effect = side_effect

        X = smbo.rng.rand(10, 2)
        Y = smbo.rng.rand(10, 1)

        challengers = smbo.choose_next(X, Y)
        x = [c for c in challengers]

        self.assertEqual(smbo.model.train.call_count, 1)
        self.assertEqual(len(x), 2020)
        num_random_search = 0
        num_local_search = 0
        for i in range(0, 2020, 2):
            # print(x[i].origin)
            self.assertIsInstance(x[i], Configuration)
            if 'Random Search (sorted)' in x[i].origin:
                num_random_search += 1
            elif 'Local Search' in x[i].origin:
                num_local_search += 1
        # number of local search configs has to be least 10
        # since x can have duplicates
        # which can be associated with the local search
        self.assertGreaterEqual(num_local_search, 10)
        for i in range(1, 2020, 2):
            self.assertIsInstance(x[i], Configuration)
            self.assertEqual(x[i].origin, 'Random Search')

    def test_choose_next_empty_X(self):
        smbo = SMAC(self.scenario, rng=1).solver
        smbo.acquisition_func._compute = mock.Mock(spec=RandomForestWithInstances)
        smbo._get_next_by_random_search = mock.Mock(spec=smbo._get_next_by_random_search)
        smbo._get_next_by_random_search.return_value = [[0, 0], [0, 1], [0, 2]]

        X = np.zeros((0, 2))
        Y = np.zeros((0, 1))

        x = smbo.choose_next(X, Y)
        self.assertEqual(x, [0, 1, 2])
        self.assertEqual(smbo._get_next_by_random_search.call_count, 1)
        self.assertEqual(smbo.acquisition_func._compute.call_count, 0)

    def test_choose_next_empty_X_2(self):
        smbo = SMAC(self.scenario, rng=1).solver

        X = np.zeros((0, 2))
        Y = np.zeros((0, 1))

        x = smbo.choose_next(X, Y)
        self.assertEqual(len(x), 1)
        self.assertIsInstance(x[0], Configuration)

    @mock.patch('smac.optimizer.smbo.convert_configurations_to_array')
    @mock.patch.object(EI, '__call__')
    @mock.patch.object(ConfigurationSpace, 'sample_configuration')
    def test_get_next_by_random_search_sorted(self,
                                              patch_sample,
                                              patch_ei,
                                              patch_impute):
        values = (10, 1, 9, 2, 8, 3, 7, 4, 6, 5)
        patch_sample.return_value = [ConfigurationMock(i) for i in values]
        patch_ei.return_value = np.array([[_] for _ in values], dtype=float)
        patch_impute.side_effect = lambda l: values
        smbo = SMAC(self.scenario, rng=1).solver
        rval = smbo._get_next_by_random_search(10, True)
        self.assertEqual(len(rval), 10)
        for i in range(10):
            self.assertIsInstance(rval[i][1], ConfigurationMock)
            self.assertEqual(rval[i][1].value, 10 - i)
            self.assertEqual(rval[i][0], 10 - i)
            self.assertEqual(rval[i][1].origin, 'Random Search (sorted)')

        # Check that config.get_array works as desired and imputation is used
        #  in between
        np.testing.assert_allclose(patch_ei.call_args[0][0],
                                   np.array(values, dtype=float))

    @mock.patch.object(ConfigurationSpace, 'sample_configuration')
    def test_get_next_by_random_search(self, patch):
        def side_effect(size):
            return [ConfigurationMock()] * size
        patch.side_effect = side_effect
        smbo = SMAC(self.scenario, rng=1).solver
        rval = smbo._get_next_by_random_search(10, False)
        self.assertEqual(len(rval), 10)
        for i in range(10):
            self.assertIsInstance(rval[i][1], ConfigurationMock)
            self.assertEqual(rval[i][1].origin, 'Random Search')
            self.assertEqual(rval[i][0], 0)

    @mock.patch.object(LocalSearch, 'maximize')
    def test_get_next_by_local_search(self, patch):
        # Without known incumbent
        class SideEffect(object):
            def __init__(self):
                self.call_number = 0

            def __call__(self, *args, **kwargs):
                rval = 9 - self.call_number
                self.call_number += 1
                return (ConfigurationMock(rval), [rval])

        patch.side_effect = SideEffect()
        smbo = SMAC(self.scenario, rng=1).solver
        rand_confs = smbo.config_space.sample_configuration(size=9)
        rval = smbo._get_next_by_local_search(init_points=rand_confs)
        self.assertEqual(len(rval), 9)
        self.assertEqual(patch.call_count, 9)
        for i in range(9):
            self.assertIsInstance(rval[i][1], ConfigurationMock)
            self.assertEqual(rval[i][1].value, 9 - i)
            self.assertEqual(rval[i][0], 9 - i)
            self.assertEqual(rval[i][1].origin, 'Local Search')

        # With known incumbent
        patch.side_effect = SideEffect()
        smbo.incumbent = 'Incumbent'
        rval = smbo._get_next_by_local_search(init_points=[smbo.incumbent]+rand_confs)
        self.assertEqual(len(rval), 10)
        self.assertEqual(patch.call_count, 19)
        # Only the first local search in each iteration starts from the
        # incumbent
        self.assertEqual(patch.call_args_list[9][0][0], 'Incumbent')
        for i in range(10):
            self.assertEqual(rval[i][1].origin, 'Local Search')

    @mock.patch.object(SingleConfigInitialDesign, 'run')
    def test_abort_on_initial_design(self, patch):
        def target(x):
            return 5
        patch.side_effect = FirstRunCrashedException()
        scen = Scenario({'cs': test_helpers.get_branin_config_space(),
                         'run_obj': 'quality', 'output_dir': '',
                         'abort_on_first_run_crash': 1})
        smbo = SMAC(scen, tae_runner=target, rng=1).solver
        self.assertRaises(FirstRunCrashedException, smbo.run)

    def test_intensification_percentage(self):
        def target(x):
            return 5
        def get_smbo(intensification_perc):
            """ Return SMBO with intensification_percentage. """
            scen = Scenario({'cs': test_helpers.get_branin_config_space(),
                             'run_obj': 'quality', 'output_dir': '',
                             'intensification_percentage' : intensification_perc})
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


if __name__ == "__main__":
    unittest.main()
