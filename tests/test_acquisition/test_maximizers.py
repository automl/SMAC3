from __future__ import annotations

import os
import unittest
import unittest.mock
import pytest
from typing import Any

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, Integer, Categorical, Float, EqualsCondition, InCondition
from ConfigSpace.hyperparameters import (
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    NormalFloatHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.model.random_forest.random_forest import RandomForest
from scipy.spatial.distance import euclidean

from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import pcs
from smac.acquisition.maximizers import (
    LocalAndSortedPriorRandomSearch,
    LocalSearch,
    RandomSearch,
)
from smac.acquisition.functions import EI
from smac.runhistory.runhistory import RunHistory
from smac.runner.abstract_runner import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class ConfigurationMock(object):
    def __init__(self, value=None):
        self.value = value

    def get_array(self):
        return [self.value]


@pytest.fixture
def configspace_branin() -> ConfigurationSpace:
    """Returns the branin configspace."""
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Float("x", (-5, 10)))
    cs.add_hyperparameter(Float("y", (0, 15)))
    return cs


def rosenbrock_4d(cfg):
    x1 = cfg["x1"]
    x2 = cfg["x2"]
    x3 = cfg["x3"]
    x4 = cfg["x4"]

    val = (
        100 * (x2 - x1**2) ** 2
        + (x1 - 1) ** 2
        + 100 * (x3 - x2**2) ** 2
        + (x2 - 1) ** 2
        + 100 * (x4 - x3**2) ** 2
        + (x3 - 1) ** 2
    )

    return val


class MockEI(EI):
    def __init__(self, values: list, **kwargs):
        super().__init__(**kwargs)
        self.values = values

    def __call__(self, configurations: list) -> np.ndarray:
        return np.array([[_] for _ in self.values], dtype=float)


class MockConfigurationSpace(object):
    def sample_configuration(self, size: int = 1) -> ConfigurationMock:
        if size == 1:
            ret = ConfigurationMock()
        else:
            ret = [ConfigurationMock()] * size

        return ret


class MockRandomSearch(RandomSearch):
    def _maximize(self, *args, **kwargs):
        return [(0, 0)]


class MethodCallLogger(object):
    def __init__(self, meth):
        self.meth = meth
        self.call_count = 0

    def __call__(self, *args):
        self.call_count += 1
        return self.meth(*args)


def test_ei_maximization_challenger_list_callback():
    values = (10, 1, 9, 2, 8, 3, 7, 4, 6, 5)
    cs = MockConfigurationSpace()
    ei = MockEI(values=values)
    rs = MockRandomSearch(configspace=cs, acquisition_function=ei)
    rs._maximize = MethodCallLogger(rs._maximize)

    rval = rs.maximize(
        previous_configs=[],
        n_points=10,
    )
    assert rs._maximize.call_count == 0
    next(rval)
    assert rs._maximize.call_count == 1

    random_design = unittest.mock.Mock()
    random_design.check.side_effect = [True, False, False, False]
    rs._maximize = unittest.mock.Mock()
    rs._maximize.return_value = [(0, 0), (1, 1)]

    rval = rs.maximize(
        previous_configs=[],
        n_points=10,
        random_design=random_design,
    )
    assert rs._maximize.call_count == 0

    # The first configuration is chosen at random (see the random_configuration_chooser mock)
    conf = next(rval)
    assert isinstance(conf, ConfigurationMock)
    assert rs._maximize.call_count == 0

    # The 2nd configuration triggers the call to the callback (see the random_configuration_chooser mock)
    conf = next(rval)
    assert rs._maximize.call_count == 1
    assert conf == 0

    # The 3rd configuration doesn't trigger the callback any more
    conf = next(rval)
    assert rs._maximize.call_count == 1
    assert conf == 1

    with pytest.raises(StopIteration):
        next(rval)


def test_ei_maximization_get_next_by_random_search():
    # def side_effect(size):
    #         return [ConfigurationMock()] * size

    # patch.side_effect = side_effect
    cs = MockConfigurationSpace()
    ei = EI(None)
    rs = RandomSearch(configspace=cs, acquisition_function=ei)
    rval = rs._maximize(previous_configs=[], n_points=10, _sorted=False)
    assert len(rval) == 10
    for i in range(10):
        assert isinstance(rval[i][1], ConfigurationMock)
        assert rval[i][1].origin == "Random Search"
        assert rval[i][0] == 0


def test_get_next_by_random_search():
    cs = ConfigurationSpace()
    ei = EI(None)
    rs = RandomSearch(configspace=cs, acquisition_function=ei)
    rval = rs._maximize(previous_configs=[], n_points=10, _sorted=False)
    assert len(rval) == 10
    for i in range(10):
        assert isinstance(rval[i][1], Configuration)
        assert rval[i][1].origin == "Random Search"
        assert rval[i][0] == 0


# --------------------------------------------------------------
# TestLocalSearch
# --------------------------------------------------------------


@pytest.fixture
def configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)

    a = Float("a", (0, 1), default=0.5)
    b = Float("b", (0, 1), default=0.5)
    c = Float("c", (0, 1), default=0.5)

    # Add all hyperparameters at once:
    cs.add_hyperparameters([a, b, c])

    return cs


@pytest.fixture
def model(configspace: ConfigurationSpace):
    model = RandomForest(configspace)

    np.random.seed(0)
    X = np.random.rand(100, len(configspace.get_hyperparameters()))
    y = 1 - (np.sum(X, axis=1) / len(configspace.get_hyperparameters()))
    model.train(X, y)

    return model


@pytest.fixture
def acquisition_function(model):
    ei = EI()
    ei.update(model=model, eta=0.5)

    return ei


def test_local_search(configspace):
    def acquisition_function(points):
        rval = []
        for point in points:
            opt = np.array([1, 1, 1])
            rval.append([-euclidean(point.get_array(), opt)])

        return np.array(rval)

    ls = LocalSearch(configspace, acquisition_function, max_steps=100)

    start_point = configspace.sample_configuration()
    acq_val_start_point = acquisition_function([start_point])
    acq_val_incumbent, _ = ls._search(start_point)[0]

    # Local search needs to find something that is as least as good as the
    # start point
    assert acq_val_start_point < acq_val_incumbent


def test_local_search_2(configspace, acquisition_function):
    start_points = configspace.sample_configuration(100)
    ls = LocalSearch(configspace, acquisition_function, max_steps=1000)

    start_point = start_points[0]
    _, incumbent = ls._maximize([start_point], 1, None)[0]

    assert incumbent.origin == "Local Search"
    assert start_point != incumbent

    # Check if they are sorted
    values = ls._maximize(start_points, 100, None)
    assert values[0][0] > values[-1][0]
    assert values[0][0] >= values[1][0]


"""
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

    points = ls._get_initial_points(n_points=5, runhistory=runhistory, additional_start_points=None)
    assert len(points), 10)


# --------------------------------------------------------------
# Test TestRandomSearch
# --------------------------------------------------------------


@unittest.mock.patch("smac.optimizer.acquisition.convert_configurations_to_array")
@unittest.mock.patch.object(EI, "__call__")
@unittest.mock.patch.object(ConfigurationSpace, "sample_configuration")
def test_get_next_by_random_search_sorted(self, patch_sample, patch_ei, patch_impute):
    values = (10, 1, 9, 2, 8, 3, 7, 4, 6, 5)
    patch_sample.return_value = [ConfigurationMock(i) for i in values]
    patch_ei.return_value = np.array([[_] for _ in values], dtype=float)
    patch_impute.side_effect = lambda l: values
    cs = ConfigurationSpace()
    ei = EI(None)
    rs = RandomSearch(ei, cs)
    rval = rs._maximize(runhistory=None, stats=None, n_points=10, _sorted=True)
    assert len(rval), 10)
    for i in range(10):
        self.assertIsInstance(rval[i][1], ConfigurationMock)
        assert rval[i][1].value, 10 - i)
        assert rval[i][0], 10 - i)
        assert rval[i][1].origin, "Random Search (sorted)")

    # Check that config.get_array works as desired and imputation is used
    #  in between, we therefore have to retrieve the value from the mock!
    np.testing.assert_allclose([v.value for v in patch_ei.call_args[0][0]], np.array(values, dtype=float))


@unittest.mock.patch.object(ConfigurationSpace, "sample_configuration")
def test_get_next_by_random_search(self, patch):
    def side_effect(size):
        return [ConfigurationMock()] * size

    patch.side_effect = side_effect
    cs = ConfigurationSpace()
    ei = EI(None)
    rs = RandomSearch(ei, cs)
    rval = rs._maximize(runhistory=None, stats=None, n_points=10, _sorted=False)
    assert len(rval), 10)
    for i in range(10):
        self.assertIsInstance(rval[i][1], ConfigurationMock)
        assert rval[i][1].origin, "Random Search")
        assert rval[i][0], 0)


# --------------------------------------------------------------
# Test TestLocalAndSortedPriorRandomSearch
# --------------------------------------------------------------


def setUp(self):
    seed = 1
    self.uniform_cs = ConfigurationSpace(seed=seed)
    x1 = UniformFloatHyperparameter("x1", -5, 5, default_value=5)
    x2 = UniformIntegerHyperparameter("x2", -5, 5, default_value=5)
    x3 = CategoricalHyperparameter("x3", [5, 2, 0, 1, -1, -2, 4, -3, 3, -5, -4], default_value=5)
    x4 = UniformIntegerHyperparameter("x4", -5, 5, default_value=5)
    self.uniform_cs.add_hyperparameters([x1, x2, x3, x4])

    self.prior_cs = ConfigurationSpace(seed=seed)
    x1 = NormalFloatHyperparameter("x1", lower=-5, upper=5, mu=0, sigma=1e-3, default_value=5)
    x2 = BetaIntegerHyperparameter("x2", lower=-5, upper=5, alpha=100, beta=1, default_value=5)
    x3 = CategoricalHyperparameter(
        "x3", [5, 2, 0, 1, -1, -2, 4, -3, 3, -5, -4], default_value=5, weights=[999, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    x4 = UniformIntegerHyperparameter("x4", lower=-5, upper=5, default_value=5)
    self.prior_cs.add_hyperparameters([x1, x2, x3, x4])

    self.budget_kwargs = {"max_steps": 2, "n_steps_plateau_walk": 2, "n_sls_iterations": 2}


def test_sampling_fractions(self):
    class AcquisitionFunction:
        def __call__(self, arrays):
            rval = []
            for array in arrays:
                rval.append([-rosenbrock_4d(array)])
            return np.array(rval)

    prs_0 = LocalAndSortedPriorRandomSearch(
        AcquisitionFunction(),
        self.prior_cs,
        self.uniform_cs,
        prior_sampling_fraction=0,
        **self.budget_kwargs,
    )

    prs_05 = LocalAndSortedPriorRandomSearch(
        AcquisitionFunction(),
        self.prior_cs,
        self.uniform_cs,
        prior_sampling_fraction=0.9,
        **self.budget_kwargs,
    )

    prs_1 = LocalAndSortedPriorRandomSearch(
        AcquisitionFunction(),
        self.prior_cs,
        self.uniform_cs,
        prior_sampling_fraction=1,
        **self.budget_kwargs,
    )

    res_0 = prs_0._maximize(runhistory=unittest.mock.Mock(), stats=None, n_points=10)
    res_05 = prs_05._maximize(runhistory=unittest.mock.Mock(), stats=None, n_points=10)
    res_1 = prs_1._maximize(runhistory=unittest.mock.Mock(), stats=None, n_points=10)
"""
