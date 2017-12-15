from contextlib import suppress
import sys
import unittest
import shutil

import numpy as np

from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, \
    RunHistory2EPM4LogCost, RunHistory2EPM4EIPS
from smac.scenario.scenario import Scenario
from smac.optimizer.acquisition import EI, EIPS, LogEI
from smac.tae.execute_func import ExecuteTAFuncArray
from smac.tae.execute_ta_run import  FirstRunCrashedException
from smac.utils import test_helpers
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.uncorrelated_mo_rf_with_instances import \
    UncorrelatedMultiObjectiveRandomForestWithInstances
from smac.utils.util_funcs import get_types
from smac.facade.epils_facade import EPILS
from smac.initial_design.single_config_initial_design import SingleConfigInitialDesign

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
                                  'output_dir': '',
                                  'runcount_limit':1,
                                  'deterministic': True})

    def tearDown(self):
        for i in range(20):
            with suppress(Exception):
                dirname = 'run_1' + ('.OLD' * i)
                shutil.rmtree(dirname)

    def branin(self, config):
        print(config)
        y = (config[1] - (5.1 / (4 * np.pi ** 2)) * config[0] ** 2 + 5 * config[0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(config[0]) + 10

        return y
    
    def test_epils(self):
        taf = ExecuteTAFuncArray(ta=self.branin)
        epils = EPILS(self.scenario, tae_runner=taf)
        inc = epils.optimize()
        # not enough runs available to change the inc
        self.assertEqual(inc["x"], 2.5)
        self.assertEqual(inc["y"], 7.5)


    def test_init_only_scenario_runtime(self):
        self.scenario.run_obj = 'runtime'
        self.scenario.cutoff = 300
        epils = EPILS(self.scenario).solver
        self.assertIsInstance(epils.model, RandomForestWithInstances)
        self.assertIsInstance(epils.rh2EPM, RunHistory2EPM4LogCost)
        self.assertIsInstance(epils.acquisition_func, LogEI)

    def test_init_only_scenario_quality(self):
        epils = EPILS(self.scenario).solver
        self.assertIsInstance(epils.model, RandomForestWithInstances)
        self.assertIsInstance(epils.rh2EPM, RunHistory2EPM4Cost)
        self.assertIsInstance(epils.acquisition_func, EI)

    def test_init_EIPS_as_arguments(self):
        for objective in ['runtime', 'quality']:
            self.scenario.run_obj = objective
            types, bounds = get_types(self.scenario.cs, None)
            umrfwi = UncorrelatedMultiObjectiveRandomForestWithInstances(
                ['cost', 'runtime'], types, bounds)
            eips = EIPS(umrfwi)
            rh2EPM = RunHistory2EPM4EIPS(self.scenario, 2)
            epils = EPILS(self.scenario, model=umrfwi, acquisition_function=eips,
                        runhistory2epm=rh2EPM).solver
            self.assertIs(umrfwi, epils.model)
            self.assertIs(eips, epils.acquisition_func)
            self.assertIs(rh2EPM, epils.rh2EPM)

    def test_rng(self):
        epils = EPILS(self.scenario, rng=None).solver
        self.assertIsInstance(epils.rng, np.random.RandomState)
        self.assertIsInstance(epils.num_run, int)
        epils = EPILS(self.scenario, rng=1).solver
        rng = np.random.RandomState(1)
        self.assertEqual(epils.num_run, 1)
        self.assertIsInstance(epils.rng, np.random.RandomState)
        epils = EPILS(self.scenario, rng=rng).solver
        self.assertIsInstance(epils.num_run, int)
        self.assertIs(epils.rng, rng)
        self.assertRaisesRegexp(TypeError,
                                "Unknown type <(class|type) 'str'> for argument "
                                'rng. Only accepts None, int or '
                                'np.random.RandomState',
                                EPILS, self.scenario, rng='BLA')

    @mock.patch.object(SingleConfigInitialDesign, 'run')
    def test_abort_on_initial_design(self, patch):
        def target(x):
            return 5
        patch.side_effect = FirstRunCrashedException()
        scen = Scenario({'cs': test_helpers.get_branin_config_space(),
                         'run_obj': 'quality', 'output_dir': '',
                         'abort_on_first_run_crash': 1})
        epils = EPILS(scen, tae_runner=target, rng=1).solver
        self.assertRaises(FirstRunCrashedException, epils.run)

if __name__ == "__main__":
    unittest.main()
