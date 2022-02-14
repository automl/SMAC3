import unittest.mock

from ConfigSpace import EqualsCondition
import scipy.optimize
import numpy as np
import sklearn.datasets
import sklearn.model_selection

from smac.configspace import (
    ConfigurationSpace,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    convert_configurations_to_array,
)
from smac.epm.gaussian_process import GaussianProcess
from smac.epm.gp_base_prior import HorseshoePrior, LognormalPrior

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_gp(n_dimensions, rs, noise=1e-3, normalize_y=True) -> GaussianProcess:
    from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel

    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(1e-10, 2),
        prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rs),
    )
    exp_kernel = Matern(
        np.ones([n_dimensions]),
        [(np.exp(-10), np.exp(2)) for _ in range(n_dimensions)],
        nu=2.5,
    )
    noise_kernel = WhiteKernel(
        noise_level=noise,
        noise_level_bounds=(1e-10, 2),
        prior=HorseshoePrior(scale=0.1, rng=rs),
    )
    kernel = cov_amp * exp_kernel + noise_kernel

    bounds = [(0., 1.) for _ in range(n_dimensions)]
    types = np.zeros(n_dimensions)

    configspace = ConfigurationSpace()
    for i in range(n_dimensions):
        configspace.add_hyperparameter(UniformFloatHyperparameter('x%d' % i, 0, 1))

    model = GaussianProcess(
        configspace=configspace,
        bounds=bounds,
        types=types,
        kernel=kernel,
        seed=rs.randint(low=1, high=10000),
        normalize_y=normalize_y,
        n_opt_restarts=2,
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


def get_mixed_gp(cat_dims, cont_dims, rs, noise=1e-3, normalize_y=True):
    from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel

    cat_dims = np.array(cat_dims, dtype=np.int)
    cont_dims = np.array(cont_dims, dtype=np.int)
    n_dimensions = len(cat_dims) + len(cont_dims)
    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(1e-10, 2),
        prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rs),
    )

    exp_kernel = Matern(
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
        prior=HorseshoePrior(scale=0.1, rng=rs),
    )
    kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel

    bounds = [0] * n_dimensions
    types = np.zeros(n_dimensions)
    for c in cont_dims:
        bounds[c] = (0., 1.)
    for c in cat_dims:
        types[c] = 3
        bounds[c] = (3, np.nan)

    cs = ConfigurationSpace()
    for c in cont_dims:
        cs.add_hyperparameter(UniformFloatHyperparameter('X%d' % c, 0, 1))
    for c in cat_dims:
        cs.add_hyperparameter(CategoricalHyperparameter('X%d' % c, [0, 1, 2, 3]))

    model = GaussianProcess(
        configspace=cs,
        bounds=bounds,
        types=types,
        kernel=kernel,
        seed=rs.randint(low=1, high=10000),
        normalize_y=normalize_y,
    )
    return model


class TestGP(unittest.TestCase):

    def test_predict_wrong_X_dimensions(self):
        rs = np.random.RandomState(1)

        # cont
        X, Y, n_dims = get_cont_data(rs)
        # cat
        X, Y, cat_dims, cont_dims = get_cat_data(rs)

        for model in (get_gp(n_dims, rs), get_mixed_gp(cat_dims, cont_dims, rs)):
            X = rs.rand(10)
            self.assertRaisesRegex(ValueError, "Expected 2d array, got 1d array!",
                                   model.predict, X)
            X = rs.rand(10, 10, 10)
            self.assertRaisesRegex(ValueError, "Expected 2d array, got 3d array!",
                                   model.predict, X)

            X = rs.rand(10, 5)
            self.assertRaisesRegex(ValueError, "Rows in X should have 10 entries "
                                               "but have 5!",
                                   model.predict, X)

    def test_predict(self):
        rs = np.random.RandomState(1)

        # cont
        X, Y, n_dims = get_cont_data(rs)
        # cat
        X, Y, cat_dims, cont_dims = get_cat_data(rs)

        for model in (get_gp(n_dims, rs), get_mixed_gp(cat_dims, cont_dims, rs)):
            model.train(X[:10], Y[:10])
            m_hat, v_hat = model.predict(X[10:])
            self.assertEqual(m_hat.shape, (10, 1))
            self.assertEqual(v_hat.shape, (10, 1))

    def test_train_do_optimize(self):
        # Check that do_optimize does not mess with the kernel hyperparameters given to the Gaussian process!

        rs = np.random.RandomState(1)
        X, Y, n_dims = get_cont_data(rs)

        model = get_gp(n_dims, rs)
        model._train(X[:10], Y[:10], do_optimize=False)
        theta = model.gp.kernel.theta
        theta_ = model.gp.kernel_.theta
        fixture = np.array([0.693147, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -6.907755])
        np.testing.assert_array_almost_equal(theta, fixture)
        np.testing.assert_array_almost_equal(theta_, fixture)
        np.testing.assert_array_almost_equal(theta, theta_)

        model._train(X[:10], Y[:10], do_optimize=True)
        theta = model.gp.kernel.theta
        theta_ = model.gp.kernel_.theta
        fixture = np.array([0.693147, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -6.907755])
        self.assertFalse(np.any(theta == fixture))
        self.assertFalse(np.any(theta_ == fixture))
        np.testing.assert_array_almost_equal(theta, theta_)

    @unittest.mock.patch('sklearn.gaussian_process.GaussianProcessRegressor.fit')
    def test_train_continue_on_linalg_error(self, fit_mock):
        # Check that training does not stop on a linalg error, but that uncertainty is increased!

        class Dummy:
            counter = 0

            def __call__(self, X, y):
                if self.counter >= 10:
                    return None
                else:
                    self.counter += 1
                    raise np.linalg.LinAlgError

        fit_mock.side_effect = Dummy()

        rs = np.random.RandomState(1)
        X, Y, n_dims = get_cont_data(rs)

        model = get_gp(n_dims, rs)
        fixture = np.exp(model.kernel.theta[-1])
        model._train(X[:10], Y[:10], do_optimize=False)
        self.assertAlmostEqual(np.exp(model.gp.kernel.theta[-1]), fixture + 10)

    @unittest.mock.patch('sklearn.gaussian_process.GaussianProcessRegressor.log_marginal_likelihood')
    def test_train_continue_on_linalg_error_2(self, fit_mock):
        # Check that training does not stop on a linalg error during hyperparameter optimization

        class Dummy:
            counter = 0

            def __call__(self, X, eval_gradient=True, clone_kernel=True):
                # If this is not aligned with the GP an error will be raised that None is not iterable
                if self.counter == 13:
                    return None
                else:
                    self.counter += 1
                    raise np.linalg.LinAlgError

        fit_mock.side_effect = Dummy()

        rs = np.random.RandomState(1)
        X, Y, n_dims = get_cont_data(rs)

        model = get_gp(n_dims, rs)
        _ = model.kernel.theta
        model._train(X[:10], Y[:10], do_optimize=True)
        np.testing.assert_array_almost_equal(
            model.gp.kernel.theta,
            [0.69314718, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.69314718]
        )

    @unittest.mock.patch.object(GaussianProcess, 'predict')
    def test_predict_marginalized_over_instances_no_features(self, rf_mock):
        """The GP should fall back to the regular predict() method."""
        rs = np.random.RandomState(1)

        # cont
        X, Y, n_dims = get_cont_data(rs)
        # cat
        X, Y, cat_dims, cont_dims = get_cat_data(rs)

        for ct, model in enumerate((get_gp(n_dims, rs), get_mixed_gp(cat_dims, cont_dims, rs))):
            model.train(X[:10], Y[:10])
            model.predict(X[10:])
            self.assertEqual(rf_mock.call_count, ct + 1)

    def test_predict_with_actual_values(self):
        X = np.array([
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.]], dtype=np.float64)
        y = np.array([
            [.1],
            [.2],
            [9],
            [9.2],
            [100.],
            [100.2],
            [109.],
            [109.2]], dtype=np.float64)
        rs = np.random.RandomState(1)
        model = get_gp(3, rs)
        model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

        mu_hat, var_hat = model.predict(X)
        for y_i, y_hat_i, mu_hat_i in zip(
            y.reshape((1, -1)).flatten(), mu_hat.reshape((1, -1)).flatten(), var_hat.reshape((1, -1)).flatten(),
        ):
            self.assertAlmostEqual(y_hat_i, y_i, delta=2)
            self.assertAlmostEqual(mu_hat_i, 0, delta=2)

        # Regression test that performance does not drastically decrease in the near future
        mu_hat, var_hat = model.predict(np.array([[10, 10, 10]]))
        self.assertAlmostEqual(mu_hat[0][0], 54.612500000000004)
        # There's a slight difference between my local installation and travis
        self.assertLess(abs(var_hat[0][0] - 1121.8409184001594), 2)

        # test other covariance results
        _, var_fc = model.predict(X, cov_return_type='full_cov')
        self.assertEqual(var_fc.shape, (8, 8))
        _, var_sd = model.predict(X, cov_return_type='diagonal_std')
        self.assertEqual(var_sd.shape, (8, 1))
        _, var_no = model.predict(np.array([[10, 10, 10]]), cov_return_type=None)
        self.assertIsNone(var_no)
        # check values
        _, var_fc = model.predict(np.array([[10, 10, 10]]), cov_return_type='full_cov')
        self.assertAlmostEqual(var_fc[0][0], var_hat[0][0])
        _, var_sd = model.predict(np.array([[10, 10, 10]]), cov_return_type='diagonal_std')
        self.assertAlmostEqual(var_sd[0][0] ** 2, var_hat[0][0])

    def test_gp_on_sklearn_data(self):
        X, y = sklearn.datasets.load_boston(return_X_y=True)
        # Normalize such that the bounds in get_gp (10) hold
        X = X / X.max(axis=0)
        rs = np.random.RandomState(1)
        model = get_gp(X.shape[1], rs)
        cv = sklearn.model_selection.KFold(shuffle=True, random_state=rs, n_splits=2)

        maes = [8.712875586609810299, 8.7608419489812271634]

        for i, (train_split, test_split) in enumerate(cv.split(X, y)):
            X_train = X[train_split]
            y_train = y[train_split]
            X_test = X[test_split]
            y_test = y[test_split]
            model.train(X_train, y_train)
            y_hat, mu_hat = model.predict(X_test)
            mae = np.mean(np.abs(y_hat - y_test), dtype=np.float128)
            self.assertAlmostEqual(mae, maes[i])

    def test_nll(self):
        rs = np.random.RandomState(1)
        gp = get_gp(1, rs)
        gp.train(np.array([[0], [1]]), np.array([0, 1]))
        n_above_1 = 0
        for i in range(1000):
            theta = np.array(
                [rs.uniform(1e-10, 10), rs.uniform(-10, 2), rs.uniform(-10, 1)]
            )  # Values from the default prior
            error = scipy.optimize.check_grad(lambda x: gp._nll(x)[0], lambda x: gp._nll(x)[1], theta, epsilon=1e-5)
            if error > 0.1:
                n_above_1 += 1
        self.assertLessEqual(n_above_1, 10)

    def test_sampling_shape(self):
        X = np.arange(-5, 5, 0.1).reshape((-1, 1))
        X_test = np.arange(-5.05, 5.05, 0.1).reshape((-1, 1))
        for shape in (None, (-1, 1)):

            if shape is None:
                y = np.sin(X).flatten()
            else:
                y = np.sin(X).reshape(shape)

            rng = np.random.RandomState(1)
            for gp in (
                get_gp(n_dimensions=1, rs=rng, noise=1e-10, normalize_y=False),
                get_gp(n_dimensions=1, rs=rng, noise=1e-10, normalize_y=True),
            ):
                gp._train(X, y)
                func = gp.sample_functions(X_test=X_test, n_funcs=1)
                self.assertEqual(func.shape, (101, 1), msg=shape)
                func = gp.sample_functions(X_test=X_test, n_funcs=2)
                self.assertEqual(func.shape, (101, 2))

    def test_normalization(self):
        X = np.arange(-5, 5, 0.1).reshape((-1, 1))
        X_test = np.arange(-5.05, 5.05, 0.1).reshape((-1, 1))
        y = np.sin(X)
        rng = np.random.RandomState(1)
        gp = get_gp(n_dimensions=1, rs=rng, noise=1e-10, normalize_y=False)
        gp._train(X, y, do_optimize=False)
        mu_hat, var_hat = gp.predict(X_test)
        gp_norm = get_gp(n_dimensions=1, rs=rng, noise=1e-10, normalize_y=True)
        gp_norm._train(X, y, do_optimize=False)
        mu_hat_prime, var_hat_prime = gp_norm.predict(X_test)
        np.testing.assert_array_almost_equal(mu_hat, mu_hat_prime, decimal=4)
        np.testing.assert_array_almost_equal(var_hat, var_hat_prime, decimal=4)

        func = gp.sample_functions(X_test=X_test, n_funcs=2)
        func_prime = gp_norm.sample_functions(X_test=X_test, n_funcs=2)
        np.testing.assert_array_almost_equal(func, func_prime, decimal=1)

    def test_impute_inactive_hyperparameters(self):
        cs = ConfigurationSpace()
        a = cs.add_hyperparameter(CategoricalHyperparameter('a', [0, 1]))
        b = cs.add_hyperparameter(CategoricalHyperparameter('b', [0, 1]))
        c = cs.add_hyperparameter(UniformFloatHyperparameter('c', 0, 1))
        cs.add_condition(EqualsCondition(b, a, 1))
        cs.add_condition(EqualsCondition(c, a, 0))
        cs.seed(1)

        configs = cs.sample_configuration(size=100)
        config_array = convert_configurations_to_array(configs)
        for line in config_array:
            if line[0] == 0:
                self.assertTrue(np.isnan(line[1]))
            elif line[0] == 1:
                self.assertTrue(np.isnan(line[2]))

        gp = get_gp(3, np.random.RandomState(1))
        config_array = gp._impute_inactive(config_array)
        for line in config_array:
            if line[0] == 0:
                self.assertEqual(line[1], -1)
            elif line[0] == 1:
                self.assertEqual(line[2], -1)
