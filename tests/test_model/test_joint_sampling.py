from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac.model.random_forest.random_forest import RandomForest

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


def _configspace(n_dimensions: int) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)
    cs.add([Float(f"x{i}", (0.0, 1.0)) for i in range(n_dimensions)])

    return cs


def _training_data(n_points: int = 40, n_dimensions: int = 3):
    rng = np.random.RandomState(0)
    X = rng.rand(n_points, n_dimensions)
    y = np.sin(3 * X.sum(axis=1)).reshape(-1, 1)

    return X, y


@pytest.fixture
def forest() -> RandomForest:
    X, y = _training_data()
    model = RandomForest(configspace=_configspace(3), n_trees=50, seed=0)
    model.train(X, y)

    return model


@pytest.fixture
def gaussian_process():
    # The Gaussian process SMAC actually ships for black-box optimization.
    from smac import Scenario
    from smac.facade.blackbox_facade import BlackBoxFacade

    X, y = _training_data()
    scenario = Scenario(_configspace(3), n_trials=10, deterministic=True)
    model = BlackBoxFacade.get_model(scenario)
    model.train(X, y)

    return model


@pytest.fixture
def test_points() -> np.ndarray:
    rng = np.random.RandomState(1)
    X = rng.rand(10, 3)

    # A repeated point must come out perfectly correlated with its twin.
    return np.vstack([X, X[0]])


def test_abstract_model_default_sampling_is_independent(forest, test_points):
    """Checks that the fallback joint sampler declares itself as uncorrelated.

    Draws from the base implementation directly on a forest and asserts that a duplicated point is
    not correlated with its twin, which is the limitation `supports_joint_samples` warns about.
    """
    from smac.model.abstract_model import AbstractModel

    samples = AbstractModel.sample_joint(forest, test_points, 2000, np.random.default_rng(0))
    correlation = np.corrcoef(samples[:, 0], samples[:, -1])[0, 1]

    assert samples.shape == (2000, len(test_points))
    assert abs(correlation) < 0.1


def test_forest_reports_joint_support(forest):
    """Checks that a forest without instance features advertises joint sampling.

    Reads the capability flags and asserts joint sampling is offered while conditioning is not,
    since a forest cannot absorb a pseudo-observation in closed form.
    """
    assert forest.supports_joint_samples
    assert not forest.supports_conditioning


def test_forest_samples_match_predicted_marginals(forest, test_points):
    """Checks that the forest's joint samples reproduce the variance the forest already reports.

    Draws 20000 samples and compares their per-point mean and standard deviation against
    `predict_marginalized`, which must agree because both are computed from the same per-tree
    predictions.
    """
    samples = forest.sample_joint(test_points, 20000, np.random.default_rng(0))
    mean, var = forest.predict_marginalized(test_points)

    assert samples.shape == (20000, len(test_points))
    assert np.allclose(samples.mean(axis=0), mean[:, 0], atol=0.02)
    assert np.allclose(samples.std(axis=0), np.sqrt(var[:, 0]), rtol=0.1, atol=0.01)


def test_forest_samples_are_correlated(forest, test_points):
    """Checks that the forest's joint samples carry correlation between points.

    Asserts a duplicated point is perfectly correlated with its twin, the property that lets a batch
    selector tell redundant candidates apart.
    """
    samples = forest.sample_joint(test_points, 4000, np.random.default_rng(0))
    correlation = np.corrcoef(samples[:, 0], samples[:, -1])[0, 1]

    assert correlation == pytest.approx(1.0, abs=1e-6)


def test_forest_with_instance_features_falls_back(test_points):
    """Checks that instance features disable the correlated sampler rather than corrupting it.

    Trains a forest with instance features, where predictions are marginalized over instances and
    the per-tree structure no longer applies, and asserts the model reports no joint support.
    """
    rng = np.random.RandomState(0)
    model = RandomForest(
        configspace=_configspace(3),
        instance_features={"i0": [0.1], "i1": [0.7]},
        n_trees=10,
        seed=0,
    )
    X = np.hstack([rng.rand(20, 3), rng.choice([0.1, 0.7], size=(20, 1))])
    model.train(X, rng.rand(20, 1))

    assert not model.supports_joint_samples


def test_gaussian_process_samples_match_predicted_marginals(gaussian_process, test_points):
    """Checks that the Gaussian process joint sampler agrees with its own posterior.

    Draws 4000 samples and compares the per-point mean against `predict_marginalized`, and asserts a
    duplicated point is perfectly correlated with its twin.
    """
    samples = gaussian_process.sample_joint(test_points, 4000, np.random.default_rng(0))
    mean, _ = gaussian_process.predict_marginalized(test_points)

    assert samples.shape == (4000, len(test_points))
    assert np.allclose(samples.mean(axis=0), mean[:, 0], atol=0.05)
    assert np.corrcoef(samples[:, 0], samples[:, -1])[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_gaussian_process_conditioning_reduces_uncertainty(gaussian_process, test_points):
    """Checks that conditioning on a pseudo-observation collapses the variance at that point.

    Conditions on the posterior mean at one point and asserts the variance there drops sharply while
    the mean stays put, which is what makes a fantasized point repel the rest of the batch.
    """
    x = test_points[3:4]
    mean_before, var_before = gaussian_process.predict(x)

    conditioned = gaussian_process.condition(x, np.array([mean_before[0, 0]]))
    mean_after, var_after = conditioned.predict(x)

    assert var_after[0, 0] < var_before[0, 0] / 10
    assert mean_after[0, 0] == pytest.approx(mean_before[0, 0], abs=1e-6)


def test_gaussian_process_conditioning_leaves_the_original_alone(gaussian_process, test_points):
    """Checks that conditioning returns a new model instead of mutating the caller's.

    Conditions on a point and asserts the original still predicts what it did before, still holds
    its original training set, and that both models share the same kernel hyperparameters.
    """
    x = test_points[3:4]
    mean_before, var_before = gaussian_process.predict(x)
    n_train_before = gaussian_process._gp.X_train_.shape[0]

    conditioned = gaussian_process.condition(x, np.array([0.0]))
    mean_after, var_after = gaussian_process.predict(x)

    assert np.allclose(mean_before, mean_after)
    assert np.allclose(var_before, var_after)
    assert gaussian_process._gp.X_train_.shape[0] == n_train_before
    assert conditioned._gp.X_train_.shape[0] == n_train_before + 1
    assert np.allclose(gaussian_process._hypers, conditioned._hypers)


def test_conditioning_is_refused_by_the_forest(forest, test_points):
    """Checks that a forest rejects conditioning instead of silently ignoring it.

    Calls `condition` on a forest and asserts NotImplementedError, so that a selector relying on
    pseudo-observations cannot quietly build a batch of identical points.
    """
    with pytest.raises(NotImplementedError):
        forest.condition(test_points[:1], np.array([0.0]))
