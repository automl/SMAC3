from unittest.mock import patch

import numpy as np
import pytest
import scipy.optimize
import sklearn.datasets
import sklearn.model_selection
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    EqualsCondition,
    UniformFloatHyperparameter,
)

from smac.model.gaussian_process.gaussian_process import GaussianProcess
from smac.model.gaussian_process.priors import HorseshoePrior, LogNormalPrior
from smac.utils.configspace import convert_configurations_to_array

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_gp(n_dimensions, seed, noise=1e-3, normalize_y=True) -> GaussianProcess:
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
    )

    noise_kernel = WhiteKernel(
        noise_level=noise,
        noise_level_bounds=(1e-10, 2),
        prior=HorseshoePrior(scale=0.1, seed=seed),
    )

    kernel = cov_amp * exp_kernel + noise_kernel

    configspace = ConfigurationSpace()
    for i in range(n_dimensions):
        configspace.add_hyperparameter(UniformFloatHyperparameter("x%d" % i, 0, 1))

    rs = np.random.RandomState(seed)

    model = GaussianProcess(
        configspace=configspace,
        kernel=kernel,
        n_restarts=2,
        normalize_y=normalize_y,
        seed=rs.randint(low=1, high=10000),
    )
    return model


def get_cont_data(rs):
    X = rs.rand(20, 10)
    Y = rs.rand(10, 1)
    n_dims = 10

    return X, Y, n_dims


def get_cat_data(rs):
    X_cont = rs.rand(20, 5)
    X_cat = rs.randint(low=0, high=3, size=(20, 5))
    X = np.concatenate([X_cat, X_cont], axis=1)
    Y = rs.rand(10, 1)
    cat_dims = [0, 1, 2, 3, 4]
    cont_dims = [5, 6, 7, 8, 9]

    return X, Y, cat_dims, cont_dims


def get_mixed_gp(cat_dims, cont_dims, seed, noise=1e-3, normalize_y=True):
    from smac.model.gaussian_process.kernels import (
        ConstantKernel,
        HammingKernel,
        MaternKernel,
        WhiteKernel,
    )

    cat_dims = np.array(cat_dims, dtype=int)
    cont_dims = np.array(cont_dims, dtype=int)

    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(1e-10, 2),
        prior=LogNormalPrior(mean=0.0, sigma=1.0, seed=seed),
    )

    exp_kernel = MaternKernel(
        np.ones([len(cont_dims)]),
        [(np.exp(-10), np.exp(2)) for _ in range(len(cont_dims))],
        nu=2.5,
        operate_on=cont_dims,
    )

    ham_kernel = HammingKernel(
        np.ones([len(cat_dims)]),
        [(np.exp(-10), np.exp(2)) for _ in range(len(cat_dims))],
        operate_on=cat_dims,
    )

    noise_kernel = WhiteKernel(
        noise_level=noise,
        noise_level_bounds=(1e-10, 2),
        prior=HorseshoePrior(scale=0.1, seed=seed),
    )

    kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel

    cs = ConfigurationSpace()
    for c in cont_dims:
        cs.add_hyperparameter(UniformFloatHyperparameter("X%d" % c, 0, 1))
    for c in cat_dims:
        cs.add_hyperparameter(CategoricalHyperparameter("X%d" % c, [0, 1, 2, 3]))

    rs = np.random.RandomState(seed)

    model = GaussianProcess(
        configspace=cs,
        kernel=kernel,
        normalize_y=normalize_y,
        seed=rs.randint(low=1, high=10000),
    )

    return model


def test_predict_wrong_X_dimensions():
    seed = 1
    rs = np.random.RandomState(seed)

    # cont
    X, Y, n_dims = get_cont_data(rs)
    # cat
    X, Y, cat_dims, cont_dims = get_cat_data(rs)

    for model in (get_gp(n_dims, seed), get_mixed_gp(cat_dims, cont_dims, seed=1)):
        X = rs.rand(10)
        with pytest.raises(ValueError, match="Expected 2d array.*"):
            model.predict(X)

        X = rs.rand(10, 10, 10)
        with pytest.raises(ValueError, match="Expected 2d array.*"):
            model.predict(X)

        X = rs.rand(10, 5)
        with pytest.raises(ValueError, match="Feature mismatch:.*"):
            model.predict(X)


def test_predict():
    seed = 1
    rs = np.random.RandomState(seed)

    # cont
    X, Y, n_dims = get_cont_data(rs)
    # cat
    X, Y, cat_dims, cont_dims = get_cat_data(rs)

    for model in (get_gp(n_dims, seed), get_mixed_gp(cat_dims, cont_dims, seed)):
        model.train(X[:10], Y[:10])
        m_hat, v_hat = model.predict(X[10:])
        assert m_hat.shape == (10, 1)
        assert v_hat.shape == (10, 1)


def test_train_do_optimize():
    # Check that do_optimize does not mess with the kernel hyperparameters given to the Gaussian process!
    seed = 1
    rs = np.random.RandomState(seed)
    X, Y, n_dims = get_cont_data(rs)

    model = get_gp(n_dims, seed)
    model._train(X[:10], Y[:10], optimize_hyperparameters=False)
    theta = model._kernel.theta
    theta_ = model._kernel.theta
    fixture = np.array([0.693147, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.907755])
    np.testing.assert_array_almost_equal(theta, fixture)
    np.testing.assert_array_almost_equal(theta_, fixture)
    np.testing.assert_array_almost_equal(theta, theta_)

    model._train(X[:10], Y[:10], optimize_hyperparameters=True)
    theta = model._kernel.theta
    theta_ = model._kernel.theta
    fixture = np.array([0.693147, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.907755])
    assert not np.any(theta == fixture)
    assert not np.any(theta_ == fixture)
    np.testing.assert_array_almost_equal(theta, theta_)


def test_train_continue_on_linalg_error():
    """Checks that training does not stop on a linalg error, but that uncertainty is increased!"""

    class Dummy:
        counter = 0

        def __call__(self, X, y):
            if self.counter >= 10:
                return None

            self.counter += 1
            raise np.linalg.LinAlgError

    with patch.object(sklearn.gaussian_process.GaussianProcessRegressor, "fit", Dummy().__call__):
        seed = 1
        rs = np.random.RandomState(seed)
        X, Y, n_dims = get_cont_data(rs)

        model = get_gp(n_dims, seed)
        fixture = np.exp(model._kernel.theta[-1])
        model._train(X[:10], Y[:10], optimize_hyperparameters=False)
        assert pytest.approx(np.exp(model._kernel.theta[-1])) == fixture + 10


def test_train_continue_on_linalg_error_2():
    # Check that training does not stop on a linalg error during hyperparameter optimization

    class Dummy:
        counter = 0

        def __call__(self, X, eval_gradient=True, clone_kernel=True):
            # If this is not aligned with the GP an error will be raised that None is not iterable
            if self.counter == 13:
                return None

            self.counter += 1
            raise np.linalg.LinAlgError

    with patch.object(sklearn.gaussian_process.GaussianProcessRegressor, "log_marginal_likelihood", Dummy().__call__):

        seed = 1
        rs = np.random.RandomState(seed)
        X, Y, n_dims = get_cont_data(rs)

        model = get_gp(n_dims, seed)
        _ = model._kernel.theta
        model._train(X[:10], Y[:10], optimize_hyperparameters=True)
        np.testing.assert_array_almost_equal(
            model._kernel.theta,
            [0.69314718, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.69314718],
        )


def test_predict_marginalized_over_instances_no_features():
    """The GP should fall back to the regular predict() method."""

    class Dummy:
        counter = 0

        def __call__(self, X, Y=None):
            self.counter += 1

    dummy = Dummy()
    with patch.object(GaussianProcess, "predict", dummy.__call__):

        seed = 1
        rs = np.random.RandomState(seed)

        # cont
        X, Y, n_dims = get_cont_data(rs)
        # cat
        X, Y, cat_dims, cont_dims = get_cat_data(rs)

        for ct, model in enumerate((get_gp(n_dims, seed), get_mixed_gp(cat_dims, cont_dims, seed))):
            model.train(X[:10], Y[:10])
            model.predict(X[10:])
            assert dummy.counter == ct + 1


def test_predict_with_actual_values():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([[0.1], [0.2], [9], [9.2], [100.0], [100.2], [109.0], [109.2]], dtype=np.float64)
    seed = 1
    rs = np.random.RandomState(seed)
    model = get_gp(3, seed)
    model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

    mu_hat, var_hat = model.predict(X)
    for y_i, y_hat_i, mu_hat_i in zip(
        y.reshape((1, -1)).flatten(),
        mu_hat.reshape((1, -1)).flatten(),
        var_hat.reshape((1, -1)).flatten(),
    ):
        assert pytest.approx(y_hat_i, 0.0001) == y_i
        assert round(mu_hat_i) == 0

    # Regression test that performance does not drastically decrease in the near future
    mu_hat, var_hat = model.predict(np.array([[10, 10, 10]]))
    assert pytest.approx(mu_hat[0][0]) == 54.612500000000004
    # There's a slight difference between my local installation and travis
    # assert abs(var_hat[0][0] - 1121.8409184001594) < 2

    # test other covariance results
    _, var_fc = model.predict(X, covariance_type="full")
    assert var_fc.shape == (8, 8)

    _, var_sd = model.predict(X, covariance_type="diagonal")
    assert var_sd.shape == (8, 1)

    _, var_no = model.predict(np.array([[10, 10, 10]]), covariance_type=None)
    assert var_no is None

    # check values
    _, var_fc = model.predict(np.array([[10, 10, 10]]), covariance_type="full")
    assert pytest.approx(var_fc[0][0]) == var_hat[0][0]

    _, var_sd = model.predict(np.array([[10, 10, 10]]), covariance_type="std")
    assert pytest.approx(var_sd[0][0] ** 2) == var_hat[0][0]


# def test_gp_on_sklearn_data():
#     X, y = sklearn.datasets.load_boston(return_X_y=True)
#     # Normalize such that the bounds in get_gp (10) hold
#     X = X / X.max(axis=0)
#     seed = 1
#     rs = np.random.RandomState(seed)
#     model = get_gp(X.shape[1], seed)
#     cv = sklearn.model_selection.KFold(shuffle=True, random_state=rs, n_splits=2)

#     maes = [8.886514052493349145, 8.868823494592284107]

#     for i, (train_split, test_split) in enumerate(cv.split(X, y)):
#         X_train = X[train_split]
#         y_train = y[train_split]
#         X_test = X[test_split]
#         y_test = y[test_split]
#         model.train(X_train, y_train)
#         y_hat, mu_hat = model.predict(X_test)
#         mae = np.mean(np.abs(y_hat - y_test), dtype=float)
#         assert pytest.approx(mae) == maes[i]


def test_nll():
    seed = 1
    rs = np.random.RandomState(seed)
    gp = get_gp(seed, seed)
    gp.train(np.array([[0], [1]]), np.array([0, 1]))
    n_above_1 = 0
    for i in range(1000):
        theta = np.array(
            [rs.uniform(1e-10, 10), rs.uniform(-10, 2), rs.uniform(-10, 1)]
        )  # Values from the default prior
        error = scipy.optimize.check_grad(lambda x: gp._nll(x)[0], lambda x: gp._nll(x)[1], theta, epsilon=1e-5)
        if error > 0.1:
            n_above_1 += 1
    assert n_above_1 < 10


def test_sampling_shape():
    X = np.arange(-5, 5, 0.1).reshape((-1, 1))
    X_test = np.arange(-5.05, 5.05, 0.1).reshape((-1, 1))
    for shape in (None, (-1, 1)):

        if shape is None:
            y = np.sin(X).flatten()
        else:
            y = np.sin(X).reshape(shape)

        seed = 1
        rng = np.random.RandomState(seed)
        for gp in (
            get_gp(n_dimensions=1, seed=seed, noise=1e-10, normalize_y=False),
            get_gp(n_dimensions=1, seed=seed, noise=1e-10, normalize_y=True),
        ):
            gp._train(X, y)
            func = gp.sample_functions(X_test=X_test, n_funcs=1)
            assert func.shape == (101, 1)
            func = gp.sample_functions(X_test=X_test, n_funcs=2)
            assert func.shape == (101, 2)


def test_normalization():
    X = np.arange(-5, 5, 0.1).reshape((-1, 1))
    X_test = np.arange(-5.05, 5.05, 0.1).reshape((-1, 1))
    y = np.sin(X)
    seed = 1

    gp = get_gp(n_dimensions=1, seed=seed, noise=1e-10, normalize_y=False)
    gp._train(X, y, optimize_hyperparameters=False)
    mu_hat, var_hat = gp.predict(X_test)

    gp_norm = get_gp(n_dimensions=1, seed=seed, noise=1e-10, normalize_y=True)
    gp_norm._train(X, y, optimize_hyperparameters=False)
    mu_hat_prime, var_hat_prime = gp_norm.predict(X_test)

    np.testing.assert_array_almost_equal(mu_hat, mu_hat_prime, decimal=4)
    np.testing.assert_array_almost_equal(var_hat, var_hat_prime, decimal=4)

    func = gp.sample_functions(X_test=X_test, n_funcs=2)
    func_prime = gp_norm.sample_functions(X_test=X_test, n_funcs=2)
    np.testing.assert_array_almost_equal(func, func_prime, decimal=1)


"""
def test_impute_inactive_hyperparameters():
    cs = ConfigurationSpace()
    a = cs.add_hyperparameter(CategoricalHyperparameter("a", [0, 1]))
    b = cs.add_hyperparameter(CategoricalHyperparameter("b", [0, 1]))
    c = cs.add_hyperparameter(UniformFloatHyperparameter("c", 0, 1))
    cs.add_condition(EqualsCondition(b, a, 1))
    cs.add_condition(EqualsCondition(c, a, 0))
    cs.seed(1)

    configs = cs.sample_configuration(size=100)
    config_array = convert_configurations_to_array(configs)
    for line in config_array:
        if line[0] == 0:
            assert np.isnan(line[1])
        elif line[0] == 1:
            assert np.isnan(line[2])

    gp = get_gp(3, 1)
    config_array = gp._impute_inactive(config_array)
    for line in config_array:
        if line[0] == 0:
            assert line[1] == -1
        elif line[0] == 1:
            assert line[2] == -1
"""
