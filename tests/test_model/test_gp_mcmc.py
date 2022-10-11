from unittest.mock import patch

import numpy as np
import pytest
import sklearn.datasets
import sklearn.model_selection
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from smac.model.gaussian_process.mcmc_gaussian_process import MCMCGaussianProcess
from smac.model.gaussian_process.priors import HorseshoePrior, LogNormalPrior

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_gp(n_dimensions, seed, noise=1e-3, normalize_y=True, average_samples=False, n_iter=50):
    from smac.model.gaussian_process.kernels import (
        ConstantKernel,
        MaternKernel,
        WhiteKernel,
    )

    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(1e-10, 2),
        prior=LogNormalPrior(mean=0.0, sigma=1.0, seed=seed),
    )

    exp_kernel = MaternKernel(
        np.ones([n_dimensions]),
        [(np.exp(-10), np.exp(2)) for _ in range(n_dimensions)],
        nu=2.5,
        prior=None,
    )

    noise_kernel = WhiteKernel(
        noise_level=noise,
        noise_level_bounds=(1e-10, 2),
        prior=HorseshoePrior(scale=0.1, seed=seed),
    )

    kernel = cov_amp * exp_kernel + noise_kernel

    n_mcmc_walkers = 3 * len(kernel.theta)
    if n_mcmc_walkers % 2 == 1:
        n_mcmc_walkers += 1

    configspace = ConfigurationSpace()
    for i in range(n_dimensions):
        configspace.add_hyperparameter(UniformFloatHyperparameter("x%d" % i, 0, 1))

    rs = np.random.RandomState(seed)
    model = MCMCGaussianProcess(
        configspace=configspace,
        kernel=kernel,
        n_mcmc_walkers=n_mcmc_walkers,
        chain_length=n_iter,
        burning_steps=n_iter,
        normalize_y=normalize_y,
        seed=rs.randint(low=1, high=10000),
        mcmc_sampler="emcee",
        average_samples=average_samples,
    )
    return model


def test_predict_wrong_X_dimensions():
    seed = 1
    rs = np.random.RandomState(seed)
    model = get_gp(10, seed)

    X = rs.rand(10)
    with pytest.raises(ValueError, match="Expected 2d array.*"):
        model.predict(X)

    X = rs.rand(10, 10, 10)
    with pytest.raises(ValueError, match="Expected 2d array.*"):
        model.predict(X)

    X = rs.rand(10, 5)
    with pytest.raises(ValueError, match="Feature mismatch.*"):
        model.predict(X)


def test_gp_train():
    seed = 1
    rs = np.random.RandomState(seed)
    X = rs.rand(20, 10)
    Y = rs.rand(10, 1)

    fixture = np.array([0.693147, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.907755])

    model = get_gp(10, seed)
    np.testing.assert_array_almost_equal(model._kernel.theta, fixture)
    model.train(X[:10], Y[:10])
    assert len(model.models) == 36

    for base_model in model.models:
        theta = base_model._gp.kernel.theta
        theta_ = base_model._gp.kernel_.theta

        # Test that the kernels of the base GP are actually changed!
        np.testing.assert_array_almost_equal(theta, theta_)
        assert not np.any(theta == fixture)
        assert not np.any(theta_ == fixture)


def test_gp_train_posterior_mean():
    seed = 1
    rs = np.random.RandomState(seed)
    X = rs.rand(20, 10)
    Y = rs.rand(10, 1)

    fixture = np.array([0.693147, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.907755])

    model = get_gp(10, seed, average_samples=True)
    np.testing.assert_array_almost_equal(model._kernel.theta, fixture)
    model.train(X[:10], Y[:10])

    for base_model in model.models:
        theta = base_model._gp.kernel.theta
        theta_ = base_model._gp.kernel_.theta
        # Test that the kernels of the base GP are actually changed!
        np.testing.assert_array_almost_equal(theta, theta_)
        assert not np.any(theta == fixture)
        assert not np.any(theta_ == fixture)

    assert len(model.models) == 1


def test_predict():
    seed = 1
    rs = np.random.RandomState(seed)
    X = rs.rand(20, 10)
    Y = rs.rand(10, 1)
    model = get_gp(10, seed)
    model.train(X[:10], Y[:10])
    m_hat, v_hat = model.predict(X[10:])
    assert m_hat.shape == (10, 1)
    assert v_hat.shape == (10, 1)


def test_predict_marginalized_over_instances_no_features():
    """The GP should fall back to the regular predict() method."""

    class Dummy:
        counter = 0

        def __call__(self, X, Y=None):
            self.counter += 1

    dummy = Dummy()
    with patch.object(MCMCGaussianProcess, "predict", dummy.__call__):
        seed = 1
        rs = np.random.RandomState(seed)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 1)
        model = get_gp(10, seed)
        model.train(X[:10], Y[:10])
        model.predict(X[10:])
        assert dummy.counter == 1


# def test_predict_with_actual_values():
#     X = np.array(
#         [
#             [0.0, 0.0, 0.0],
#             [0.0, 0.0, 1.0],
#             [0.0, 1.0, 0.0],
#             [0.0, 1.0, 1.0],
#             [1.0, 0.0, 0.0],
#             [1.0, 0.0, 1.0],
#             [1.0, 1.0, 0.0],
#             [1.0, 1.0, 1.0],
#         ],
#         dtype=np.float64,
#     )
#     y = np.array([[0.1], [0.2], [9], [9.2], [100.0], [100.2], [109.0], [109.2]], dtype=np.float64)
#     seed = 1
#     model = get_gp(3, seed, noise=1e-10, n_iter=200)
#     model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

#     y_hat, var_hat = model.predict(X)
#     for y_i, y_hat_i, var_hat_i in zip(
#         y.reshape((1, -1)).flatten(),
#         y_hat.reshape((1, -1)).flatten(),
#         var_hat.reshape((1, -1)).flatten(),
#     ):
#         # Chain length too short to get excellent predictions, apparently there's a lot of predictive variance
#         np.testing.assert_almost_equal(y_i, y_hat_i, decimal=1)
#         np.testing.assert_almost_equal(var_hat_i, 0, decimal=500)

#     # Regression test that performance does not drastically decrease in the near future
#     y_hat, var_hat = model.predict(np.array([[10, 10, 10]]))
#     np.testing.assert_almost_equal(y_hat[0][0], 54.613410745846785, decimal=0.1)
#     # Massive variance due to internally used law of total variances, also a massive difference locally and on
#     # travis-ci
#     assert abs(var_hat[0][0]) - 3700 <= 200


# def test_gp_on_sklearn_data():
#     X, y = sklearn.datasets.load_boston(return_X_y=True)
#     # Normalize such that the bounds in get_gp hold
#     X = X / X.max(axis=0)
#     seed = 1
#     rs = np.random.RandomState(seed)
#     model = get_gp(X.shape[1], seed, noise=1e-10, normalize_y=True)
#     cv = sklearn.model_selection.KFold(shuffle=True, random_state=rs, n_splits=2)

#     maes = [7.1383486992745653755, 7.453042020795519766]

#     for i, (train_split, test_split) in enumerate(cv.split(X, y)):
#         X_train = X[train_split]
#         y_train = y[train_split]
#         X_test = X[test_split]
#         y_test = y[test_split]
#         model.train(X_train, y_train)
#         y_hat, mu_hat = model.predict(X_test)
#         mae = np.mean(np.abs(y_hat - y_test), dtype=float)

#         np.testing.assert_almost_equal(mae, maes[i])


def test_normalization():
    X = np.arange(-5, 5, 0.1).reshape((-1, 1))
    X_test = np.arange(-5.05, 5.05, 0.1).reshape((-1, 1))
    y = np.sin(X)
    seed = 1
    gp = get_gp(n_dimensions=1, seed=seed, noise=1e-10, normalize_y=False)
    gp._train(X, y, optimize_hyperparameters=False)
    assert not gp.models[0]._normalize_y
    assert not hasattr(gp.models[0], "mean_y_")

    mu_hat, var_hat = gp.predict(X_test)
    gp_norm = get_gp(n_dimensions=1, seed=seed, noise=1e-10, normalize_y=True)
    gp_norm._train(X, y, optimize_hyperparameters=False)
    assert gp_norm.models[0]._normalize_y
    assert hasattr(gp_norm.models[0], "mean_y_")

    mu_hat_prime, var_hat_prime = gp_norm.predict(X_test)
    np.testing.assert_array_almost_equal(mu_hat, mu_hat_prime, decimal=4)
    np.testing.assert_array_almost_equal(var_hat, var_hat_prime, decimal=4)
