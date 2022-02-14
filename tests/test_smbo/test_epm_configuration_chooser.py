import shutil
import unittest
from unittest import mock

import numpy as np
from ConfigSpace import Configuration

from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.facade.smac_ac_facade import SMAC4AC
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.utils import test_helpers
from smac.tae import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class ConfigurationMock(object):
    def __init__(self, value=None):
        self.value = value

    def get_array(self):
        return [self.value]


class TestEPMChooser(unittest.TestCase):

    def setUp(self):
        self.scenario = Scenario({'cs': test_helpers.get_branin_config_space(),
                                  'run_obj': 'quality',
                                  'output_dir': 'data-test_epmchooser'})
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

    def test_choose_next(self):
        seed = 42
        config = self.scenario.cs.sample_configuration()
        rh = RunHistory()
        rh.add(config, 10, 10, StatusType.SUCCESS)

        smbo = SMAC4AC(self.scenario, rng=seed, runhistory=rh).solver

        x = next(smbo.epm_chooser.choose_next()).get_array()
        self.assertEqual(x.shape, (2,))

    def test_choose_next_budget(self):
        seed = 42
        config = self.scenario.cs.sample_configuration()
        rh = RunHistory()
        rh.add(config=config, cost=10, time=10, instance_id=None,
               seed=1, budget=1, additional_info=None, status=StatusType.SUCCESS)

        smbo = SMAC4AC(self.scenario, rng=seed, runhistory=rh).solver
        smbo.epm_chooser.min_samples_model = 2

        # There is no model, so it returns a single random configuration
        x = smbo.epm_chooser.choose_next()
        self.assertEqual(len(x), 1)
        self.assertEqual(next(x).origin, "Random Search")

    def test_choose_next_higher_budget(self):
        seed = 42
        config = self.scenario.cs.sample_configuration
        rh = RunHistory()
        rh.add(config=config(), cost=1, time=10, instance_id=None,
               seed=1, budget=1, additional_info=None, status=StatusType.SUCCESS)
        rh.add(config=config(), cost=2, time=10, instance_id=None,
               seed=1, budget=2, additional_info=None, status=StatusType.SUCCESS)
        rh.add(config=config(), cost=3, time=10, instance_id=None,
               seed=1, budget=2, additional_info=None, status=StatusType.SUCCESS)
        rh.add(config=config(), cost=4, time=10, instance_id=None,
               seed=1, budget=3, additional_info=None, status=StatusType.SUCCESS)

        smbo = SMAC4AC(self.scenario, rng=seed, runhistory=rh).solver
        smbo.epm_chooser.min_samples_model = 2

        # Return two configurations evaluated with budget==2
        X, Y, X_configurations = smbo.epm_chooser._collect_data_to_train_model()
        self.assertListEqual(list(Y.flatten()), [2, 3])
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(X_configurations.shape[0], 2)

    def test_choose_next_w_empty_rh(self):
        seed = 42
        smbo = SMAC4AC(self.scenario, rng=seed).solver
        smbo.runhistory = RunHistory()

        # should return random search configuration
        x = smbo.epm_chooser.choose_next(incumbent_value=0.0)
        self.assertEqual(len(x), 1)
        next_one = next(x)
        self.assertEqual(next_one.get_array().shape, (2,))
        self.assertEqual(next_one.origin, 'Random Search')

    def test_choose_next_empty_X(self):
        epm_chooser = SMAC4AC(self.scenario, rng=1).solver.epm_chooser
        epm_chooser.acquisition_func._compute = mock.Mock(
            spec=RandomForestWithInstances
        )
        epm_chooser._random_search.maximize = mock.Mock(
            spec=epm_chooser._random_search.maximize
        )
        epm_chooser._random_search.maximize.return_value = [0, 1, 2]

        x = epm_chooser.choose_next()
        self.assertEqual(x, [0, 1, 2])
        self.assertEqual(epm_chooser._random_search.maximize.call_count, 1)
        self.assertEqual(epm_chooser.acquisition_func._compute.call_count, 0)

    def test_choose_next_empty_X_2(self):
        epm_chooser = SMAC4AC(self.scenario, rng=1).solver.epm_chooser

        challengers = epm_chooser.choose_next()
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

        seed = 42
        incumbent = self.scenario.cs.get_default_configuration()
        rh = RunHistory()
        rh.add(incumbent, 10, 10, StatusType.SUCCESS)
        epm_chooser = SMAC4AC(self.scenario, rng=seed, runhistory=rh).solver.epm_chooser

        epm_chooser.model = mock.Mock(spec=RandomForestWithInstances)
        epm_chooser.model.predict_marginalized_over_instances.side_effect = side_effect_predict
        epm_chooser.acquisition_func._compute = mock.Mock(spec=RandomForestWithInstances)
        epm_chooser.acquisition_func._compute.side_effect = side_effect
        epm_chooser.incumbent = incumbent

        challengers = epm_chooser.choose_next()
        # Convert challenger list (a generator) to a real list
        challengers = [c for c in challengers]

        self.assertEqual(epm_chooser.model.train.call_count, 1)

        # For each configuration it is randomly sampled whether to take it from the list of challengers or to sample it
        # completely at random. Therefore, it is not guaranteed to obtain twice the number of configurations selected
        # by EI.
        self.assertEqual(len(challengers), 10198)
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
                raise ValueError((c.origin, 'Local Search' == c.origin, type('Local Search'), type(c.origin)))

        self.assertEqual(num_local_search, 11)
        self.assertEqual(num_random_search_sorted, 5000)
        self.assertEqual(num_random_search, 5187)

    def test_choose_next_3(self):
        # Test with ten configurations in the runhistory
        def side_effect(X):
            return np.mean(X, axis=1).reshape((-1, 1))

        def side_effect_predict(X):
            m, v = np.ones((X.shape[0], 1)), None
            return m, v

        epm_chooser = SMAC4AC(self.scenario, rng=1).solver.epm_chooser
        epm_chooser.incumbent = self.scenario.cs.sample_configuration()
        previous_configs = [epm_chooser.incumbent] + [self.scenario.cs.sample_configuration() for _ in range(0, 20)]
        epm_chooser.runhistory = RunHistory()
        for i, config in enumerate(previous_configs):
            epm_chooser.runhistory.add(config, i, 10, StatusType.SUCCESS)
        epm_chooser.model = mock.Mock(spec=RandomForestWithInstances)
        epm_chooser.model.predict_marginalized_over_instances.side_effect = side_effect_predict
        epm_chooser.acquisition_func._compute = mock.Mock(spec=RandomForestWithInstances)
        epm_chooser.acquisition_func._compute.side_effect = side_effect

        challengers = epm_chooser.choose_next()
        # Convert challenger list (a generator) to a real list
        challengers = [c for c in challengers]

        self.assertEqual(epm_chooser.model.train.call_count, 1)

        # For each configuration it is randomly sampled whether to take it from the list of challengers or to sample it
        # completely at random. Therefore, it is not guaranteed to obtain twice the number of configurations selected
        # by EI
        self.assertEqual(len(challengers), 9986)
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

        self.assertEqual(num_local_search, 26)
        self.assertEqual(num_random_search_sorted, 5000)
        self.assertEqual(num_random_search, 4960)


if __name__ == "__main__":
    unittest.main()
