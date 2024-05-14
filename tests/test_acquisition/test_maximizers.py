from __future__ import annotations

from typing import Any

import os
import unittest
import unittest.mock

import numpy as np
import pytest
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from ConfigSpace.hyperparameters import (
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    NormalFloatHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.read_and_write import pcs
from scipy.spatial.distance import euclidean

from smac.acquisition.function import EI
from smac.acquisition.maximizer import (
    DifferentialEvolution,
    LocalAndSortedRandomSearch,
    LocalSearch,
    RandomSearch,
)
from smac.model.random_forest.random_forest import RandomForest
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
    cs = MockConfigurationSpace()
    ei = EI(None)
    rs = RandomSearch(configspace=cs, acquisition_function=ei)
    rval = rs._maximize(previous_configs=[], n_points=10, _sorted=False)
    assert len(rval) == 10
    for i in range(10):
        assert isinstance(rval[i][1], ConfigurationMock)
        assert rval[i][1].origin == "Acquisition Function Maximizer: Random Search"
        assert rval[i][0] == 0


def test_get_next_by_random_search():
    cs = ConfigurationSpace()
    ei = EI(None)
    rs = RandomSearch(configspace=cs, acquisition_function=ei)
    rval = rs._maximize(previous_configs=[], n_points=10, _sorted=False)
    assert len(rval) == 10
    for i in range(10):
        assert isinstance(rval[i][1], Configuration)
        assert rval[i][1].origin == "Acquisition Function Maximizer: Random Search"
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

    assert incumbent.origin == "Acquisition Function Maximizer: Local Search"
    assert start_point != incumbent

    # Check if they are sorted
    values = ls._maximize(start_points, 100, None)
    assert values[0][0] > values[-1][0]
    assert values[0][0] >= values[1][0]


def test_get_initial_points_moo(configspace):
    class Model:
        def predict_marginalized(self, X):
            return X, X

    class AcquisitionFunction:

        model = Model()

        def __call__(self, X):
            return np.array([x.get_array().sum() for x in X]).reshape((-1, 1))

    ls = LocalSearch(
        configspace=configspace,
        acquisition_function=AcquisitionFunction(),
        n_steps_plateau_walk=10,
        max_steps=np.inf,
    )

    random_configs = configspace.sample_configuration(size=100)
    points = ls._get_initial_points(random_configs, n_points=5, additional_start_points=None)
    assert len(points) == 5


# --------------------------------------------------------------
# TestRandomSearch
# --------------------------------------------------------------


def test_random_search(configspace, acquisition_function):
    start_points = configspace.sample_configuration(100)
    rs = RandomSearch(configspace, acquisition_function, challengers=1000)

    start_point = start_points[0]
    _, incumbent = rs._maximize([start_point], 1, None)[0]

    assert incumbent.origin == "Acquisition Function Maximizer: Random Search"
    assert start_point != incumbent

    # Check if they all are 0 (because we don't get a maximize value for random search)
    values = rs._maximize(start_points, 100)
    assert values[0][0] == 0
    assert values[0][0] == values[-1][0]
    assert values[0][0] == values[1][0]
    assert all([v[0] == 0 for v in values])


def test_random_search_sorted(configspace, acquisition_function):
    start_points = configspace.sample_configuration(100)
    rs = RandomSearch(configspace, acquisition_function, challengers=1000)

    start_point = start_points[0]
    _, incumbent = rs._maximize([start_point], 1, _sorted=True)[0]

    assert incumbent.origin == "Acquisition Function Maximizer: Random Search (sorted)"
    assert start_point != incumbent

    # Check if they are higher than 0 (because we sorted them)
    values = rs._maximize(start_points, 100, _sorted=True)
    assert all([v[0] > 0 for v in values])


# --------------------------------------------------------------
# TestLocalAndRandomSearch
# --------------------------------------------------------------


def test_local_and_random_search(configspace, acquisition_function):
    start_points = configspace.sample_configuration(100)
    rs = LocalAndSortedRandomSearch(configspace, acquisition_function, challengers=1000)
    assert rs.acquisition_function == acquisition_function
    assert rs._random_search.acquisition_function == acquisition_function
    assert rs._local_search.acquisition_function == acquisition_function

    values = rs._maximize(start_points, 100)
    config_origins = []
    v_old = np.inf
    for (v, config) in values:
        config_origins += [config.origin]
        if isinstance(v, np.ndarray):
            v = float(v[0])

        assert v_old >= v
        v_old = v

    assert "Acquisition Function Maximizer: Local Search" in config_origins


# --------------------------------------------------------------
# TestLocalAndSortedPriorRandomSearch
# --------------------------------------------------------------


@pytest.fixture
def configspace_rosenbrock():
    seed = 1
    uniform_cs = ConfigurationSpace(seed=seed)
    x1 = UniformFloatHyperparameter("x1", -5, 5, default_value=5)
    x2 = UniformIntegerHyperparameter("x2", -5, 5, default_value=5)
    x3 = CategoricalHyperparameter("x3", [5, 2, 0, 1, -1, -2, 4, -3, 3, -5, -4], default_value=5)
    x4 = UniformIntegerHyperparameter("x4", -5, 5, default_value=5)
    uniform_cs.add_hyperparameters([x1, x2, x3, x4])

    return uniform_cs


@pytest.fixture
def configspace_prior():
    seed = 1

    prior_cs = ConfigurationSpace(seed=seed)
    x1 = NormalFloatHyperparameter("x1", lower=-5, upper=5, mu=0, sigma=1e-3, default_value=5)
    x2 = BetaIntegerHyperparameter("x2", lower=-5, upper=5, alpha=100, beta=1, default_value=5)
    x3 = CategoricalHyperparameter(
        "x3", [5, 2, 0, 1, -1, -2, 4, -3, 3, -5, -4], default_value=5, weights=[999, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    x4 = UniformIntegerHyperparameter("x4", lower=-5, upper=5, default_value=5)
    prior_cs.add_hyperparameters([x1, x2, x3, x4])

    return prior_cs


def test_sampling_fractions(configspace_rosenbrock, configspace_prior):
    class AcquisitionFunction:
        def __call__(self, arrays):
            rval = []
            for array in arrays:
                rval.append([-rosenbrock_4d(array)])
            return np.array(rval)

    budget_kwargs = {"max_steps": 2, "n_steps_plateau_walk": 2, "local_search_iterations": 2}

    prs_0 = LocalAndSortedRandomSearch(
        configspace=configspace_prior,
        uniform_configspace=configspace_rosenbrock,
        acquisition_function=AcquisitionFunction(),
        prior_sampling_fraction=0,
        **budget_kwargs,
    )

    prs_05 = LocalAndSortedRandomSearch(
        configspace=configspace_prior,
        uniform_configspace=configspace_rosenbrock,
        acquisition_function=AcquisitionFunction(),
        prior_sampling_fraction=0.9,
        **budget_kwargs,
    )

    prs_1 = LocalAndSortedRandomSearch(
        configspace=configspace_prior,
        uniform_configspace=configspace_rosenbrock,
        acquisition_function=AcquisitionFunction(),
        prior_sampling_fraction=1,
        **budget_kwargs,
    )

    prs_0._maximize(previous_configs=[], n_points=10)
    prs_05._maximize(previous_configs=[], n_points=10)
    prs_1._maximize(previous_configs=[], n_points=10)


# --------------------------------------------------------------
# TestDifferentialEvolution
# --------------------------------------------------------------


def test_differential_evolution(configspace, acquisition_function):
    start_points = configspace.sample_configuration(100)
    rs = DifferentialEvolution(configspace, acquisition_function, challengers=1000)

    values = rs._maximize(start_points, 1)
    values[0][1].origin == "Acquisition Function Maximizer: Differential Evolution"
