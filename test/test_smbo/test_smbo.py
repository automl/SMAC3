import shutil
import sys
import time
import unittest
from unittest import mock

import numpy as np

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.utils.constants import MAXINT
from smac.configspace import ConfigurationSpace
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.facade.smac_ac_facade import SMAC4AC
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.intensification.abstract_racer import RunInfoIntent
from smac.optimizer.acquisition import EI, LogEI
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, RunHistory2EPM4LogCost
from smac.runhistory.runhistory import RunInfo, RunValue, RunKey
from smac.scenario.scenario import Scenario
from smac.tae import FirstRunCrashedException, StatusType
from smac.tae.execute_func import ExecuteTAFuncArray
from smac.utils import test_helpers
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator


def target(x, seed, instance):
    """A target function for dummy testing of TA
    perform x^2 for easy result calculations in checks.
    """
    # Return x[i] (with brackets) so we pass the value, not the
    # np array element
    return x[0] ** 2, {'key': seed, 'instance': instance}


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

    @mock.patch('smac.tae.execute_func.ExecuteTAFuncDict._call_ta')
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

    @mock.patch('smac.tae.execute_func.AbstractTAFunc.run')
    def test_abort_on_runner(self, patch):
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

    @mock.patch('smac.tae.execute_func.AbstractTAFunc.run')
    def test_stop_smbo(self, patch):
        def target(x):
            return 5

        # should raise an error if abort_on_first_run_crash is True
        patch.return_value = StatusType.STOP, 0.5, 0.5, {}
        scen = Scenario({'cs': test_helpers.get_branin_config_space(),
                         'run_obj': 'quality', 'output_dir': 'data-test_smbo-abort',
                         'abort_on_first_run_crash': True})
        self.output_dirs.append(scen.output_dir)
        smbo = SMAC4AC(scen, tae_runner=target, rng=1)
        self.assertFalse(smbo.solver._stop)
        smbo.optimize()
        self.assertEqual(len(smbo.runhistory.data), 1)
        self.assertEqual(list(smbo.runhistory.data.values())[0].status, StatusType.RUNNING)
        self.assertTrue(smbo.solver._stop)

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
        This test checks the intensification time bound is updated in subsequent iterations as long as
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
            def __init__(self, intensifier, get_next_run):
                self.intensifier = intensifier
                self.get_next_run = get_next_run
                self.counter = 0

            def __call__(self, *args, **kwargs):
                self.counter += 1
                if self.counter % 4 == 0:
                    self.intensifier.num_run = 0
                return self.get_next_run(*args, **kwargs)

        solver.intensifier.get_next_run = unittest.mock.Mock(
            side_effect=SideEffect(solver.intensifier, solver.intensifier.get_next_run))

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

    @unittest.skipIf(sys.version_info < (3, 6), 'distributed requires Python >=3.6')
    def test_ta_integration_to_smbo(self):
        """
        In SMBO. 3 objects need to actively comunicate:
            -> stats
            -> epm
            -> runhistory

        This method makes sure that executed jobs are properly registered
        in the above objects

        It uses n_workers to test parallel and serial implementations!!
        """

        for n_workers in range(1, 2):
            # We create a controlled setting, in which we optimize x^2
            # This will allow us to make sure every component act as expected

            # FIRST: config space
            cs = ConfigurationSpace()
            cs.add_hyperparameter(UniformFloatHyperparameter('x', -10.0, 10.0))
            smac = SMAC4HPO(
                scenario=Scenario({
                    'n_workers': n_workers,
                    'cs': cs,
                    'runcount_limit': 5,
                    'run_obj': 'quality',
                    "deterministic": "true",
                    "initial_incumbent": "DEFAULT",
                    'output_dir': 'data-test_smbo'
                }),
                tae_runner=ExecuteTAFuncArray,
                tae_runner_kwargs={'ta': target},
            )

            # Register output dir for deletion
            self.output_dirs.append(smac.output_dir)

            smbo = smac.solver

            # SECOND: Intensifier that tracks configs
            all_configs = []

            def mock_get_next_run(**kwargs):
                config = cs.sample_configuration()
                all_configs.append(config)
                return (RunInfoIntent.RUN, RunInfo(
                    config=config, instance=time.time() % 10,
                    instance_specific={}, seed=0,
                    cutoff=None, capped=False, budget=0.0
                ))
            intensifier = unittest.mock.Mock()
            intensifier.num_run = 0
            intensifier.process_results.return_value = (0.0, 0.0)
            intensifier.get_next_run = mock_get_next_run
            smac.solver.intensifier = intensifier

            # THIRD: Run in this controlled setting
            smbo.run()

            # FOURTH: Checks

            # Make sure all configs where launched
            self.assertEqual(len(all_configs), 5)

            # Run history
            for k, v in smbo.runhistory.data.items():

                # All configuration should be successful
                self.assertEqual(v.status, StatusType.SUCCESS)

                # The value should be the square version of the config
                # The runhistory has  config_ids = {config: int}
                # The k here is {config_id: int}. We search for the actual config
                # by inverse searching this runhistory.config dict
                config = list(smbo.runhistory.config_ids.keys())[
                    list(smbo.runhistory.config_ids.values()).index(k.config_id)
                ]

                self.assertEqual(v.cost,
                                 config.get('x')**2
                                 )

            # No config is lost in the config history
            self.assertCountEqual(smbo.runhistory.config_ids.keys(), all_configs)

            # Stats!
            # We do not exceed the number of target algorithm runs
            self.assertEqual(smbo.stats.submitted_ta_runs, len(all_configs))
            self.assertEqual(smbo.stats.finished_ta_runs, len(all_configs))

            # No config is lost
            self.assertEqual(smbo.stats.n_configs, len(all_configs))

            # The EPM can access all points. This is something that
            # also relies on the runhistory
            X, Y, X_config = smbo.epm_chooser._collect_data_to_train_model()
            self.assertEqual(X.shape[0], len(all_configs))

    @unittest.skipIf(sys.version_info < (3, 6), 'distributed requires Python >=3.6')
    def test_incorporate_run_results(self):
        """
        Makes sure that a configuration that as launched and tagged as running
        is updated properly in the run history, when the job is completed
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformFloatHyperparameter('x', -10.0, 10.0))
        smac = SMAC4HPO(
            scenario=Scenario({
                'n_workers': 2,
                'cs': cs,
                'runcount_limit': 5,
                'run_obj': 'quality',
                "deterministic": "true",
                "initial_incumbent": "DEFAULT",
                'output_dir': 'data-test_smbo'
            }),
            tae_runner=ExecuteTAFuncArray,
            tae_runner_kwargs={'ta': target},
        )

        # Register output dir for deletion
        self.output_dirs.append(smac.output_dir)

        smbo = smac.solver

        # Mock the fact that we launched a run
        # Here the run value is the default assumption in
        # case of the run not finishing
        config = cs.sample_configuration()
        instance = 5
        run_info = RunInfo(
            config=config, instance=instance,
            instance_specific={}, seed=0,
            cutoff=None, capped=False, budget=0.0
        )
        starttime = time.time()
        run_value = RunValue(
            cost=float(MAXINT),
            time=0.0,
            status=StatusType.RUNNING,
            starttime=starttime,
            endtime=0.0,
            additional_info=None
        )
        config_id = smbo.runhistory.add(
            config=run_info.config,
            cost=float(MAXINT),
            time=0.0,
            status=StatusType.RUNNING,
            instance_id=run_info.instance,
            seed=run_info.seed,
            budget=run_info.budget,
            starttime=starttime,
        )
        run_info.config.config_id = config_id

        # We should only have one value in the run history
        self.assertEqual(config_id, 1)
        self.assertDictEqual(
            smbo.runhistory.data,
            {RunKey(
                config_id=config_id,
                instance_id=run_info.instance,
                seed=run_info.seed,
                budget=run_info.budget
            ): run_value,
            }
        )

        # Then we test with updating the finished run

        # Mock intensifier
        smbo.intensifier = unittest.mock.Mock()
        smbo.intensifier.process_results.return_value = (config, 1.0)

        run_value = RunValue(
            cost=123,
            time=0.5,
            status=StatusType.SUCCESS,
            starttime=time.time(),
            endtime=time.time() + 1,
            additional_info={}
        )
        smbo._incorporate_run_results(run_info=run_info,
                                      result=run_value,
                                      time_left=np.inf)
        self.assertEqual(smbo.stats.n_configs, 1)
        self.assertDictEqual(
            smbo.runhistory.data,
            {RunKey(
                config_id=config_id,
                instance_id=run_info.instance,
                seed=run_info.seed,
                budget=run_info.budget
            ): run_value,
            }
        )

        # here drastically change the config, so we rely on the
        # config id handshaking
        config = cs.sample_configuration()
        run_info = RunInfo(
            config=config, instance=instance,
            instance_specific={}, seed=0,
            cutoff=None, capped=False, budget=0.0
        )
        run_info.config.config_id = config_id
        smbo._incorporate_run_results(run_info=run_info,
                                      result=run_value,
                                      time_left=np.inf)
        self.assertEqual(smbo.stats.n_configs, 1)
        self.assertDictEqual(
            smbo.runhistory.data,
            {RunKey(
                config_id=config_id,
                instance_id=run_info.instance,
                seed=run_info.seed,
                budget=run_info.budget
            ): run_value,
            }
        )


if __name__ == "__main__":
    unittest.main()
