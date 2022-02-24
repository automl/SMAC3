import unittest
import unittest.mock
import os

import numpy as np
from scipy.spatial.distance import euclidean

from smac.configspace import pcs
from smac.optimizer.acquisition import EI
from smac.optimizer.ei_optimization import LocalSearch, RandomSearch
from smac.runhistory.runhistory import RunHistory
from smac.tae import StatusType
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.utils import test_helpers

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class ConfigurationMock(object):
    def __init__(self, value=None):
        self.value = value

    def get_array(self):
        return [self.value]


def rosenbrock_4d(cfg):
    x1 = cfg["x1"]
    x2 = cfg["x2"]
    x3 = cfg["x3"]
    x4 = cfg["x4"]

    val = (100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2 + 100 * (
        x3 - x2 ** 2) ** 2 + (x2 - 1) ** 2 + 100 * (x4 - x3 ** 2) ** 2 + (x3 - 1) ** 2)

    return(val)


class TestEIMaximization(unittest.TestCase):
    @unittest.mock.patch('smac.optimizer.acquisition.convert_configurations_to_array')
    @unittest.mock.patch.object(EI, '__call__')
    @unittest.mock.patch.object(ConfigurationSpace, 'sample_configuration')
    def test_challenger_list_callback(self, patch_sample, patch_ei, patch_impute):
        values = (10, 1, 9, 2, 8, 3, 7, 4, 6, 5)
        patch_sample.return_value = ConfigurationMock(1)
        patch_ei.return_value = np.array([[_] for _ in values], dtype=float)
        patch_impute.side_effect = lambda l: values
        cs = ConfigurationSpace()
        ei = EI(None)
        rs = RandomSearch(ei, cs)
        rs._maximize = unittest.mock.Mock()
        rs._maximize.return_value = [(0, 0)]

        rval = rs.maximize(
            runhistory=None, stats=None, num_points=10,
        )
        self.assertEqual(rs._maximize.call_count, 0)
        next(rval)
        self.assertEqual(rs._maximize.call_count, 1)

        random_configuration_chooser = unittest.mock.Mock()
        random_configuration_chooser.check.side_effect = [True, False, False, False]
        rs._maximize = unittest.mock.Mock()
        rs._maximize.return_value = [(0, 0), (1, 1)]

        rval = rs.maximize(
            runhistory=None, stats=None, num_points=10, random_configuration_chooser=random_configuration_chooser,
        )
        self.assertEqual(rs._maximize.call_count, 0)
        # The first configuration is chosen at random (see the random_configuration_chooser mock)
        conf = next(rval)
        self.assertIsInstance(conf, ConfigurationMock)
        self.assertEqual(rs._maximize.call_count, 0)
        # The 2nd configuration triggers the call to the callback (see the random_configuration_chooser mock)
        conf = next(rval)
        self.assertEqual(rs._maximize.call_count, 1)
        self.assertEqual(conf, 0)
        # The 3rd configuration doesn't trigger the callback any more
        conf = next(rval)
        self.assertEqual(rs._maximize.call_count, 1)
        self.assertEqual(conf, 1)

        with self.assertRaises(StopIteration):
            next(rval)

    @unittest.mock.patch.object(ConfigurationSpace, 'sample_configuration')
    def test_get_next_by_random_search(self, patch):
        def side_effect(size):
            return [ConfigurationMock()] * size

        patch.side_effect = side_effect
        cs = ConfigurationSpace()
        ei = EI(None)
        rs = RandomSearch(ei, cs)
        rval = rs._maximize(
            runhistory=None, stats=None, num_points=10, _sorted=False
        )
        self.assertEqual(len(rval), 10)
        for i in range(10):
            self.assertIsInstance(rval[i][1], ConfigurationMock)
            self.assertEqual(rval[i][1].origin, 'Random Search')
            self.assertEqual(rval[i][0], 0)


class TestLocalSearch(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self.test_files_dir = os.path.join(current_dir, '..', 'test_files')
        seed = np.random.randint(1, 100000)
        self.cs = ConfigurationSpace(seed=seed)
        x1 = UniformFloatHyperparameter("x1", -5, 5, default_value=5)
        self.cs.add_hyperparameter(x1)
        x2 = UniformIntegerHyperparameter("x2", -5, 5, default_value=5)
        self.cs.add_hyperparameter(x2)
        x3 = CategoricalHyperparameter(
            "x3", [5, 2, 0, 1, -1, -2, 4, -3, 3, -5, -4], default_value=5)
        self.cs.add_hyperparameter(x3)
        x4 = UniformIntegerHyperparameter("x4", -5, 5, default_value=5)
        self.cs.add_hyperparameter(x4)

    def test_local_search(self):

        def acquisition_function(points):
            rval = []
            for point in points:
                opt = np.array([1, 1, 1, 1])
                rval.append([-euclidean(point.get_array(), opt)])
            return np.array(rval)

        ls = LocalSearch(acquisition_function, self.cs, max_steps=100)

        start_point = self.cs.sample_configuration()
        acq_val_start_point = acquisition_function([start_point])

        acq_val_incumbent, _ = ls._do_search(start_point)[0]

        # Local search needs to find something that is as least as good as the
        # start point
        self.assertLessEqual(acq_val_start_point, acq_val_incumbent)

    @unittest.mock.patch.object(LocalSearch, '_get_initial_points')
    def test_local_search_2(
            self,
            _get_initial_points_patch,
    ):
        pcs_file = os.path.join(self.test_files_dir, "test_local_search.pcs")
        seed = np.random.randint(1, 100000)

        runhistory = unittest.mock.Mock()
        runhistory.data = [None] * 1000

        with open(pcs_file) as fh:
            config_space = pcs.read(fh.readlines())
            config_space.seed(seed)

        def acquisition_function(points):
            return np.array([[np.count_nonzero(point.get_array())] for point in points])

        start_point = config_space.get_default_configuration()
        _get_initial_points_patch.return_value = [start_point]

        ls = LocalSearch(acquisition_function, config_space, max_steps=100000)
        # To have some data in a mock runhistory
        ls.runhistory = [None] * 1000
        acq_val_incumbent, incumbent = ls._maximize(runhistory, None, 1)[0]

        np.testing.assert_allclose(
            incumbent.get_array(),
            np.ones(len(config_space.get_hyperparameters()))
        )

    @unittest.mock.patch.object(LocalSearch, '_do_search')
    @unittest.mock.patch.object(LocalSearch, '_get_initial_points')
    def test_get_next_by_local_search(
            self,
            _get_initial_points_patch,
            patch
    ):
        # Without known incumbent
        class SideEffect(object):

            def __call__(self, *args, **kwargs):
                rval = []
                for i in range(len(args[0])):
                    rval.append((i, ConfigurationMock(i)))
                return rval

        patch.side_effect = SideEffect()
        cs = test_helpers.get_branin_config_space()
        rand_confs = cs.sample_configuration(size=9)
        _get_initial_points_patch.return_value = rand_confs
        acq_func = EI(None)

        ls = LocalSearch(acq_func, cs)

        # To have some data in a mock runhistory
        runhistory = unittest.mock.Mock()
        runhistory.data = [None] * 1000

        rval = ls._maximize(runhistory, None, 9)
        self.assertEqual(len(rval), 9)
        self.assertEqual(patch.call_count, 1)
        for i in range(9):
            self.assertIsInstance(rval[i][1], ConfigurationMock)
            self.assertEqual(rval[i][1].value, 8 - i)
            self.assertEqual(rval[i][0], 8 - i)
            self.assertEqual(rval[i][1].origin, 'Local Search')

        # Check that the known 'incumbent' is transparently passed through
        patch.side_effect = SideEffect()
        _get_initial_points_patch.return_value = ['Incumbent'] + rand_confs
        rval = ls._maximize(runhistory, None, 10)
        self.assertEqual(len(rval), 10)
        self.assertEqual(patch.call_count, 2)
        # Only the first local search in each iteration starts from the
        # incumbent
        self.assertEqual(patch.call_args_list[1][0][0][0], 'Incumbent')
        for i in range(10):
            self.assertEqual(rval[i][1].origin, 'Local Search')

    def test_local_search_finds_minimum(self):

        class AcquisitionFunction:

            model = None

            def __call__(self, arrays):
                rval = []
                for array in arrays:
                    rval.append([-rosenbrock_4d(array)])
                return np.array(rval)

        ls = LocalSearch(
            acquisition_function=AcquisitionFunction(),
            config_space=self.cs,
            n_steps_plateau_walk=10,
            max_steps=np.inf,
        )

        runhistory = RunHistory()
        self.cs.seed(1)
        random_configs = self.cs.sample_configuration(size=100)
        costs = [rosenbrock_4d(random_config) for random_config in random_configs]
        self.assertGreater(np.min(costs), 100)
        for random_config, cost in zip(random_configs, costs):
            runhistory.add(config=random_config, cost=cost, time=0, status=StatusType.SUCCESS)
        minimizer = ls.maximize(runhistory, None, 10)
        minima = [-rosenbrock_4d(m) for m in minimizer]
        self.assertGreater(minima[0], -0.05)

    def test_get_initial_points_moo(self):
        class Model:

            def predict_marginalized_over_instances(self, X):
                return X, X

        class AcquisitionFunction:

            model = Model()

            def __call__(self, X):
                return np.array([x.get_array().sum() for x in X]).reshape((-1, 1))

        ls = LocalSearch(
            acquisition_function=AcquisitionFunction(),
            config_space=self.cs,
            n_steps_plateau_walk=10,
            max_steps=np.inf,
        )

        runhistory = RunHistory()
        random_configs = self.cs.sample_configuration(size=100)
        costs = np.array([rosenbrock_4d(random_config) for random_config in random_configs])
        for random_config, cost in zip(random_configs, costs):
            runhistory.add(config=random_config, cost=cost, time=0, status=StatusType.SUCCESS)

        points = ls._get_initial_points(num_points=5, runhistory=runhistory, additional_start_points=None)
        self.assertEqual(len(points), 10)


class TestRandomSearch(unittest.TestCase):
    @unittest.mock.patch('smac.optimizer.acquisition.convert_configurations_to_array')
    @unittest.mock.patch.object(EI, '__call__')
    @unittest.mock.patch.object(ConfigurationSpace, 'sample_configuration')
    def test_get_next_by_random_search_sorted(self,
                                              patch_sample,
                                              patch_ei,
                                              patch_impute):
        values = (10, 1, 9, 2, 8, 3, 7, 4, 6, 5)
        patch_sample.return_value = [ConfigurationMock(i) for i in values]
        patch_ei.return_value = np.array([[_] for _ in values], dtype=float)
        patch_impute.side_effect = lambda l: values
        cs = ConfigurationSpace()
        ei = EI(None)
        rs = RandomSearch(ei, cs)
        rval = rs._maximize(
            runhistory=None, stats=None, num_points=10, _sorted=True
        )
        self.assertEqual(len(rval), 10)
        for i in range(10):
            self.assertIsInstance(rval[i][1], ConfigurationMock)
            self.assertEqual(rval[i][1].value, 10 - i)
            self.assertEqual(rval[i][0], 10 - i)
            self.assertEqual(rval[i][1].origin, 'Random Search (sorted)')

        # Check that config.get_array works as desired and imputation is used
        #  in between, we therefore have to retrieve the value from the mock!
        np.testing.assert_allclose([v.value for v in patch_ei.call_args[0][0]],
                                   np.array(values, dtype=float))

    @unittest.mock.patch.object(ConfigurationSpace, 'sample_configuration')
    def test_get_next_by_random_search(self, patch):
        def side_effect(size):
            return [ConfigurationMock()] * size

        patch.side_effect = side_effect
        cs = ConfigurationSpace()
        ei = EI(None)
        rs = RandomSearch(ei, cs)
        rval = rs._maximize(
            runhistory=None, stats=None, num_points=10, _sorted=False
        )
        self.assertEqual(len(rval), 10)
        for i in range(10):
            self.assertIsInstance(rval[i][1], ConfigurationMock)
            self.assertEqual(rval[i][1].origin, 'Random Search')
            self.assertEqual(rval[i][0], 0)


if __name__ == "__main__":
    unittest.main()
