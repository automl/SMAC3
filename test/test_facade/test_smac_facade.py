from contextlib import suppress
import os
import shutil
import unittest
import unittest.mock

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.util import get_one_exchange_neighbourhood
from nose.plugins.attrib import attr

from smac.configspace import ConfigurationSpace

from smac.epm.base_epm import AbstractEPM
from smac.epm.random_epm import RandomEPM
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.uncorrelated_mo_rf_with_instances import UncorrelatedMultiObjectiveRandomForestWithInstances
from smac.facade.smac_facade import SMAC
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.initial_design.initial_design import InitialDesign
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.factorial_design import FactorialInitialDesign
from smac.initial_design.sobol_design import SobolDesign
from smac.intensification.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, RunHistory2EPM4EIPS
from smac.scenario.scenario import Scenario
from smac.optimizer.acquisition import EI, AbstractAcquisitionFunction, EIPS, LCB
from smac.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, LocalSearch
from smac.optimizer.random_configuration_chooser import (
    ChooserCosineAnnealing,
    ChooserProb,
)
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.tae.execute_ta_run import ExecuteTARun
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.util_funcs import get_rng

class TestSMACFacade(unittest.TestCase):

    def setUp(self):
        self.cs = ConfigurationSpace()
        self.scenario = Scenario({'cs': self.cs, 'run_obj': 'quality',
                                  'output_dir': ''})
        self.output_dirs = []

    def tearDown(self):
        for i in range(20):
            with suppress(Exception):
                dirname = 'run_1' + ('.OLD' * i)
                shutil.rmtree(dirname)
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)

    ####################################################################################################################
    # Test that the objects are constructed correctly

    def test_pass_callable(self):
        # Check that SMAC accepts a callable as target algorithm and that it is
        # correctly wrapped with ExecuteTaFunc
        def target_algorithm(conf, inst):
            return 5
        smac = SMAC(tae_runner=target_algorithm, scenario=self.scenario)
        self.assertIsInstance(smac.solver.intensifier.tae_runner,
                              ExecuteTAFuncDict)
        self.assertIs(smac.solver.intensifier.tae_runner.ta, target_algorithm)

    def test_pass_invalid_tae_runner(self):
        self.assertRaisesRegex(
            TypeError,
            "Argument 'tae_runner' is <class 'int'>, but must be either None, a callable or an "
            "object implementing ExecuteTaRun.",
            SMAC,
            tae_runner=1,
            scenario=self.scenario,
        )

    def test_pass_tae_runner_objective(self):
        self.assertRaisesRegex(
            ValueError,
            "Objective for the target algorithm runner and the scenario must be the same, but are 'runtime' and "
            "'quality'",
            SMAC,
            tae_runner=lambda: 1,
            tae_runner_kwargs={'run_obj': 'runtime'},
            scenario=self.scenario,
        )

    def test_construct_runhistory(self):
        fixture = 'dummy'

        smbo = SMAC(self.scenario)
        self.assertIsInstance(smbo.solver.runhistory, RunHistory)
        smbo = SMAC(self.scenario, runhistory_kwargs={'aggregate_func': fixture})
        self.assertEqual(smbo.solver.runhistory.aggregate_func, fixture)
        smbo = SMAC(self.scenario, runhistory=RunHistory)
        self.assertIsInstance(smbo.solver.runhistory, RunHistory)
        rh = RunHistory(aggregate_func=None)
        smbo = SMAC(self.scenario, runhistory=rh)
        self.assertIsNotNone(smbo.solver.runhistory.aggregate_func)

    def test_construct_random_configuration_chooser(self):
        rng = np.random.RandomState(42)
        smbo = SMAC(self.scenario)
        self.assertIsInstance(smbo.solver.random_configuration_chooser, ChooserProb)
        self.assertIsNot(smbo.solver.random_configuration_chooser, rng)
        smbo = SMAC(self.scenario, rng=rng)
        self.assertIsInstance(smbo.solver.random_configuration_chooser, ChooserProb)
        self.assertIs(smbo.solver.random_configuration_chooser.rng, rng)
        smbo = SMAC(self.scenario, random_configuration_chooser_kwargs={'rng': rng})
        self.assertIsInstance(smbo.solver.random_configuration_chooser, ChooserProb)
        self.assertIs(smbo.solver.random_configuration_chooser.rng, rng)
        smbo = SMAC(self.scenario, random_configuration_chooser_kwargs={'prob': 0.1})
        self.assertIsInstance(smbo.solver.random_configuration_chooser, ChooserProb)
        self.assertEqual(smbo.solver.random_configuration_chooser.prob, 0.1)
        smbo = SMAC(
            self.scenario,
            random_configuration_chooser=ChooserCosineAnnealing,
            random_configuration_chooser_kwargs={'prob_max': 1, 'prob_min': 0.1, 'restart_iteration': 10},
        )
        self.assertIsInstance(smbo.solver.random_configuration_chooser, ChooserCosineAnnealing)
        # Check for construction failure on wrong argument
        with self.assertRaisesRegex(Exception, 'got an unexpected keyword argument'):
            SMAC(self.scenario, random_configuration_chooser_kwargs={'dummy': 0.1})

    def test_construct_epm(self):
        rng = np.random.RandomState(42)
        smbo = SMAC(self.scenario)
        self.assertIsInstance(smbo.solver.model, RandomForestWithInstances)
        smbo = SMAC(self.scenario, rng=rng)
        self.assertIsInstance(smbo.solver.model, RandomForestWithInstances)
        self.assertEqual(smbo.solver.model.seed, 1935803228)
        smbo = SMAC(self.scenario, model_kwargs={'seed': 2})
        self.assertIsInstance(smbo.solver.model, RandomForestWithInstances)
        self.assertEqual(smbo.solver.model.seed, 2)
        smbo = SMAC(self.scenario, model_kwargs={'num_trees': 20})
        self.assertIsInstance(smbo.solver.model, RandomForestWithInstances)
        self.assertEqual(smbo.solver.model.rf_opts.num_trees, 20)
        smbo = SMAC(self.scenario, model=RandomEPM, model_kwargs={'seed': 2})
        self.assertIsInstance(smbo.solver.model, RandomEPM)
        self.assertEqual(smbo.solver.model.seed, 2)
        # Check for construction failure on wrong argument
        with self.assertRaisesRegex(Exception, 'got an unexpected keyword argument'):
            SMAC(self.scenario, model_kwargs={'dummy': 0.1})

    def test_construct_acquisition_function(self):
        rng = np.random.RandomState(42)
        smbo = SMAC(self.scenario)
        self.assertIsInstance(smbo.solver.acquisition_func, EI)
        smbo = SMAC(self.scenario, rng=rng)
        self.assertIsInstance(smbo.solver.acquisition_func.model, RandomForestWithInstances)
        self.assertEqual(smbo.solver.acquisition_func.model.seed, 1935803228)
        smbo = SMAC(self.scenario, acquisition_function_kwargs={'par': 17})
        self.assertIsInstance(smbo.solver.acquisition_func, EI)
        self.assertEqual(smbo.solver.acquisition_func.par, 17)
        smbo = SMAC(self.scenario, acquisition_function=LCB, acquisition_function_kwargs={'par': 19})
        self.assertIsInstance(smbo.solver.acquisition_func, LCB)
        self.assertEqual(smbo.solver.acquisition_func.par, 19)
        # Check for construction failure on wrong argument
        with self.assertRaisesRegex(Exception, 'got an unexpected keyword argument'):
            SMAC(self.scenario, acquisition_function_kwargs={'dummy': 0.1})

    def test_construct_intensifier(self):

        class DummyIntensifier(Intensifier):
            pass

        rng = np.random.RandomState(42)
        smbo = SMAC(self.scenario)
        self.assertIsInstance(smbo.solver.intensifier, Intensifier)
        self.assertIsNot(smbo.solver.intensifier.rs, rng)
        smbo = SMAC(self.scenario, rng=rng)
        self.assertIsInstance(smbo.solver.intensifier, Intensifier)
        self.assertIs(smbo.solver.intensifier.rs, rng)
        smbo = SMAC(self.scenario, intensifier_kwargs={'maxR': 987})
        self.assertEqual(smbo.solver.intensifier.maxR, 987)
        smbo = SMAC(
            self.scenario, intensifier=DummyIntensifier, intensifier_kwargs={'maxR': 987},
        )
        self.assertIsInstance(smbo.solver.intensifier, DummyIntensifier)
        self.assertEqual(smbo.solver.intensifier.maxR, 987)
        # Check for construction failure on wrong argument
        with self.assertRaisesRegex(Exception, 'got an unexpected keyword argument'):
            SMAC(self.scenario, intensifier_kwargs={'dummy': 0.1})

    def test_construct_initial_design(self):

        rng = np.random.RandomState(42)
        smbo = SMAC(self.scenario)
        self.assertIsInstance(smbo.solver.initial_design, DefaultConfiguration)
        self.assertIsNot(smbo.solver.intensifier.rs, rng)
        smbo = SMAC(self.scenario, rng=rng)
        self.assertIsInstance(smbo.solver.intensifier, Intensifier)
        self.assertIs(smbo.solver.intensifier.rs, rng)
        smbo = SMAC(self.scenario, intensifier_kwargs={'maxR': 987})
        self.assertEqual(smbo.solver.intensifier.maxR, 987)
        smbo = SMAC(
            self.scenario,
            initial_design=InitialDesign,
            initial_design_kwargs={'configs': 'dummy'},
        )
        self.assertIsInstance(smbo.solver.initial_design, InitialDesign)
        self.assertEqual(smbo.solver.initial_design.configs, 'dummy')
        # Check for construction failure on wrong argument
        with self.assertRaisesRegex(Exception, 'got an unexpected keyword argument'):
            SMAC(self.scenario, intensifier_kwargs={'dummy': 0.1})

        for initial_incumbent_string, expected_instance in (
            ("DEFAULT", DefaultConfiguration),
            ("RANDOM", RandomConfigurations),
            ("LHD", LHDesign),
            ("FACTORIAL", FactorialInitialDesign),
            ("SOBOL", SobolDesign),
        ):
            self.scenario.initial_incumbent = initial_incumbent_string
            smbo = SMAC(self.scenario)
            self.assertIsInstance(smbo.solver.initial_design, expected_instance)

    def test_init_EIPS_as_arguments(self):
        for objective in ['runtime', 'quality']:
            self.scenario.run_obj = objective
            smbo = SMAC(
                self.scenario,
                model=UncorrelatedMultiObjectiveRandomForestWithInstances,
                model_kwargs={'target_names': ['a', 'b'], 'rf_kwargs': {'seed': 1}},
                acquisition_function=EIPS,
                runhistory2epm=RunHistory2EPM4EIPS,
            ).solver
            self.assertIsInstance(smbo.model, UncorrelatedMultiObjectiveRandomForestWithInstances)
            self.assertIsInstance(smbo.acquisition_func, EIPS)
            self.assertIsInstance(smbo.acquisition_func.model, UncorrelatedMultiObjectiveRandomForestWithInstances)
            self.assertIsInstance(smbo.rh2EPM, RunHistory2EPM4EIPS)

    ####################################################################################################################
    # Other tests...

    @unittest.mock.patch.object(SMAC, '__init__')
    def test_check_random_states(self, patch):
        patch.return_value = None
        smac = SMAC()
        smac.logger = unittest.mock.MagicMock()

        # Check some properties
        # Check whether different seeds give different random states
        _, rng_1 = get_rng(1)
        _, rng_2 = get_rng(2)
        self.assertNotEqual(sum(rng_1.get_state()[1] - rng_2.get_state()[1]), 0)

        # Check whether no seeds gives different random states
        _, rng_1 = get_rng(logger=smac.logger)
        self.assertEqual(smac.logger.debug.call_count, 1)
        _, rng_2 = get_rng(logger=smac.logger)
        self.assertEqual(smac.logger.debug.call_count, 2)

        self.assertNotEqual(sum(rng_1.get_state()[1] - rng_2.get_state()[1]), 0)

        # Check whether the same int seeds give the same random states
        _, rng_1 = get_rng(1)
        _, rng_2 = get_rng(1)
        self.assertEqual(sum(rng_1.get_state()[1] - rng_2.get_state()[1]), 0)

        # Check all execution paths
        self.assertRaisesRegex(
            TypeError,
            "Argument rng accepts only arguments of type None, int or np.random.RandomState, "
            "you provided <class 'str'>.",
            get_rng,
            rng='ABC',
        )
        self.assertRaisesRegex(
            TypeError,
            "Argument run_id accepts only arguments of type None, int or np.random.RandomState, "
            "you provided <class 'str'>.",
            get_rng,
            run_id='ABC'
        )

        run_id, rng_1 = get_rng(rng=None, run_id=None, logger=smac.logger)
        self.assertIsInstance(run_id, int)
        self.assertIsInstance(rng_1, np.random.RandomState)
        self.assertEqual(smac.logger.debug.call_count, 3)

        run_id, rng_1 = get_rng(rng=None, run_id=1, logger=smac.logger)
        self.assertEqual(run_id, 1)
        self.assertIsInstance(rng_1, np.random.RandomState)

        run_id, rng_1 = get_rng(rng=1, run_id=None, logger=smac.logger)
        self.assertEqual(run_id, 1)
        self.assertIsInstance(rng_1, np.random.RandomState)

        run_id, rng_1 = get_rng(rng=1, run_id=1337, logger=smac.logger)
        self.assertEqual(run_id, 1337)
        self.assertIsInstance(rng_1, np.random.RandomState)

        rs = np.random.RandomState(1)
        run_id, rng_1 = get_rng(rng=rs, run_id=None, logger=smac.logger)
        self.assertIsInstance(run_id, int)
        self.assertIs(rng_1, rs)

        run_id, rng_1 = get_rng(rng=rs, run_id=2505, logger=smac.logger)
        self.assertEqual(run_id, 2505)
        self.assertIs(rng_1, rs)

    @attr('slow')
    @unittest.mock.patch("smac.optimizer.ei_optimization.get_one_exchange_neighbourhood")
    def test_check_deterministic_rosenbrock(self, patch):

        # Make SMAC a bit faster
        patch.side_effect = lambda configuration, seed: get_one_exchange_neighbourhood(
            configuration=configuration,
            stdev=0.05,
            num_neighbors=2,
            seed=seed,
        )

        def rosenbrock_2d(x):
            x1 = x['x1']
            x2 = x['x2']
            val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
            return val

        def opt_rosenbrock():
            cs = ConfigurationSpace()

            cs.add_hyperparameter(UniformFloatHyperparameter("x1", -5, 5, default_value=-3))
            cs.add_hyperparameter(UniformFloatHyperparameter("x2", -5, 5, default_value=-4))

            scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                                 "runcount-limit": 50,  # maximum function evaluations
                                 "cs": cs,  # configuration space
                                 "deterministic": "true",
                                 "intensification_percentage": 0.000000001
                                 })

            smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                        tae_runner=rosenbrock_2d)
            incumbent = smac.optimize()
            return incumbent, smac.scenario.output_dir

        i1, output_dir = opt_rosenbrock()
        self.output_dirs.append(output_dir)
        x1_1 = i1.get('x1')
        x2_1 = i1.get('x2')
        i2, output_dir = opt_rosenbrock()
        self.output_dirs.append(output_dir)
        x1_2 = i2.get('x1')
        x2_2 = i2.get('x2')
        self.assertAlmostEqual(x1_1, x1_2)
        self.assertAlmostEqual(x2_1, x2_2)

    def test_get_runhistory_and_trajectory_and_tae_runner(self):
        def func(x):
            return x ** 2
        smac = SMAC(tae_runner=func, scenario=self.scenario)
        self.assertRaises(ValueError, smac.get_runhistory)
        self.assertRaises(ValueError, smac.get_trajectory)
        smac.trajectory = 'dummy'
        self.assertEqual(smac.get_trajectory(), 'dummy')
        smac.runhistory = 'dummy'
        self.assertEqual(smac.get_runhistory(), 'dummy')
        self.assertEqual(smac.get_tae_runner().ta, func)

    def test_output_structure(self):
        """Test whether output-dir is moved correctly."""
        test_scenario_dict = {
            'output_dir': 'test/test_files/scenario_test/tmp_output',
            'run_obj': 'quality',
            'cs': ConfigurationSpace()
        }
        scen1 = Scenario(test_scenario_dict)
        self.output_dirs.append(scen1.output_dir)
        smac = SMAC(scenario=scen1, run_id=1)

        self.assertEqual(smac.output_dir, os.path.join(
            test_scenario_dict['output_dir'], 'run_1'))
        self.assertTrue(os.path.isdir(smac.output_dir))

        smac2 = SMAC(scenario=scen1, run_id=1)
        self.assertTrue(os.path.isdir(smac2.output_dir + '.OLD'))

        smac3 = SMAC(scenario=scen1, run_id=1)
        self.assertTrue(os.path.isdir(smac3.output_dir + '.OLD.OLD'))

        smac4 = SMAC(scenario=scen1, run_id=2)
        self.assertEqual(smac4.output_dir, os.path.join(
            test_scenario_dict['output_dir'], 'run_2'))
        self.assertTrue(os.path.isdir(smac4.output_dir))
        self.assertFalse(os.path.isdir(smac4.output_dir + '.OLD.OLD.OLD'))

        # clean up (at least whats not cleaned up by tearDown)
        shutil.rmtree(smac.output_dir + '.OLD.OLD')
        shutil.rmtree(smac.output_dir + '.OLD')
        # This is done by teardown!
        #shutil.rmtree(smac.output_dir)
        shutil.rmtree(smac4.output_dir)

    def test_no_output(self):
        """ Test whether a scenario with "" as output really does not create an
        output. """
        test_scenario_dict = {
            'output_dir': '',
            'run_obj': 'quality',
            'cs': ConfigurationSpace()
        }
        scen1 = Scenario(test_scenario_dict)
        smac = SMAC(scenario=scen1, run_id=1)
        self.assertFalse(os.path.isdir(smac.output_dir))
