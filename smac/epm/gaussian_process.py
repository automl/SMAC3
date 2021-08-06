import logging
import typing

import numpy as np
from scipy import optimize

from smac.configspace import ConfigurationSpace
from smac.epm.base_gp import BaseModel
from smac.epm.gp_base_prior import Prior
from smac.utils.constants import VERY_SMALL_NUMBER

from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process import GaussianProcessRegressor

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


logger = logging.getLogger(__name__)


class GaussianProcess(BaseModel):
    """
    Gaussian process model.

    The GP hyperparameterŝ are obtained by optimizing the marginal log likelihood.

    This code is based on the implementation of RoBO:

    Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
    RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
    In: NIPS 2017 Bayesian Optimization Workshop

    Parameters
    ----------
    types : List[int]
        Specifies the number of categorical values of an input dimension where
        the i-th entry corresponds to the i-th input dimension. Let's say we
        have 2 dimension where the first dimension consists of 3 different
        categorical choices and the second dimension is continuous than we
        have to pass [3, 0]. Note that we count starting from 0.
    bounds : List[Tuple[float, float]]
        bounds of input dimensions: (lower, uppper) for continuous dims; (n_cat, np.nan) for categorical dims
    seed : int
        Model seed.
    kernel : george kernel object
        Specifies the kernel that is used for all Gaussian Process
    prior : prior object
        Defines a prior for the hyperparameters of the GP. Make sure that
        it implements the Prior interface.
    normalize_y : bool
        Zero mean unit variance normalization of the output values
    n_opt_restart : int
        Number of restarts for GP hyperparameter optimization
    instance_features : np.ndarray (I, K)
        Contains the K dimensional instance features of the I different instances
    pca_components : float
        Number of components to keep when using PCA to reduce dimensionality of instance features. Requires to
        set n_feats (> pca_dims).
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        types: typing.List[int],
        bounds: typing.List[typing.Tuple[float, float]],
        seed: int,
        kernel: Kernel,
        normalize_y: bool = True,
        n_opt_restarts: int = 10,
        instance_features: typing.Optional[np.ndarray] = None,
        pca_components: typing.Optional[int] = None,
    ):
        super().__init__(
            configspace=configspace,
            types=types,
            bounds=bounds,
            seed=seed,
            kernel=kernel,
            instance_features=instance_features,
            pca_components=pca_components,
        )

        self.normalize_y = normalize_y
        self.n_opt_restarts = n_opt_restarts

        self.hypers = np.empty((0, ))
        self.is_trained = False
        self._n_ll_evals = 0

        self._set_has_conditions()

    def _train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool = True) -> 'GaussianProcess':
        """
        Computes the Cholesky decomposition of the covariance of X and
        estimates the GP hyperparameters by optimizing the marginal
        loglikelihood. The prior mean of the GP is set to the empirical
        mean of X.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters of the kernel are used.
        """

        X = self._impute_inactive(X)
        if self.normalize_y:
            y = self._normalize_y(y)
        if len(y.shape) == 1:
            self.n_objectives_ = 1
        else:
            self.n_objectives_ = y.shape[1]
        if self.n_objectives_ == 1:
            y = y.flatten()

        n_tries = 10
        for i in range(n_tries):
            try:
                self.gp = self._get_gp()
                self.gp.fit(X, y)
                break
            except np.linalg.LinAlgError as e:
                if i == n_tries:
                    raise e
                # Assume that the last entry of theta is the noise
                theta = np.exp(self.kernel.theta)
                theta[-1] += 1
                self.kernel.theta = np.log(theta)

        if do_optimize:
            self._all_priors = self._get_all_priors(add_bound_priors=False)
            self.hypers = self._optimize()
            self.gp.kernel.theta = self.hypers
            self.gp.fit(X, y)
        else:
            self.hypers = self.gp.kernel.theta

        self.is_trained = True
        return self

    def _get_gp(self) -> GaussianProcessRegressor:
        return GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=False,
            optimizer=None,
            n_restarts_optimizer=-1,  # Do not use scikit-learn's optimization routine
            alpha=0,  # Governed by the kernel
            random_state=self.rng,
        )

    def _nll(self, theta: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        """
        Returns the negative marginal log likelihood (+ the prior) for
        a hyperparameter configuration theta.
        (negative because we use scipy minimize for optimization)

        Parameters
        ----------
        theta : np.ndarray(H)
            Hyperparameter vector. Note that all hyperparameter are
            on a log scale.

        Returns
        ----------
        float
            lnlikelihood + prior
        """
        self._n_ll_evals += 1

        try:
            lml, grad = self.gp.log_marginal_likelihood(theta, eval_gradient=True)
        except np.linalg.LinAlgError:
            return 1e25, np.zeros(theta.shape)

        for dim, priors in enumerate(self._all_priors):
            for prior in priors:
                lml += prior.lnprob(theta[dim])
                grad[dim] += prior.gradient(theta[dim])

        # We add a minus here because scipy is minimizing
        if not np.isfinite(lml).all() or not np.all(np.isfinite(grad)):
            return 1e25, np.zeros(theta.shape)
        else:
            return -lml, -grad

    def _optimize(self) -> np.ndarray:
        """
        Optimizes the marginal log likelihood and returns the best found
        hyperparameter configuration theta.

        Returns
        -------
        theta : np.ndarray(H)
            Hyperparameter vector that maximizes the marginal log likelihood
        """

        log_bounds = [(b[0], b[1]) for b in self.gp.kernel.bounds]

        # Start optimization from the previous hyperparameter configuration
        p0 = [self.gp.kernel.theta]
        if self.n_opt_restarts > 0:
            dim_samples = []

            prior = None  # type: typing.Optional[typing.Union[typing.List[Prior], Prior]]
            for dim, hp_bound in enumerate(log_bounds):
                prior = self._all_priors[dim]
                # Always sample from the first prior
                if isinstance(prior, list):
                    if len(prior) == 0:
                        prior = None
                    else:
                        prior = prior[0]
                prior = typing.cast(typing.Optional[Prior], prior)
                if prior is None:
                    try:
                        sample = self.rng.uniform(
                            low=hp_bound[0],
                            high=hp_bound[1],
                            size=(self.n_opt_restarts,),
                        )
                    except OverflowError:
                        raise ValueError('OverflowError while sampling from (%f, %f)' % (hp_bound[0], hp_bound[1]))
                    dim_samples.append(sample.flatten())
                else:
                    dim_samples.append(prior.sample_from_prior(self.n_opt_restarts).flatten())
            p0 += list(np.vstack(dim_samples).transpose())

        theta_star = None
        f_opt_star = np.inf
        for i, start_point in enumerate(p0):
            theta, f_opt, _ = optimize.fmin_l_bfgs_b(self._nll, start_point, bounds=log_bounds)
            if f_opt < f_opt_star:
                f_opt_star = f_opt
                theta_star = theta
        return theta_star

    def _predict(self, X_test: np.ndarray,
                 cov_return_type: typing.Optional[str] = 'diagonal_cov') \
            -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        cov_return_type: typing.Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,) or np.array(N, N) or None
            predictive variance or standard deviation

        """

        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        X_test = self._impute_inactive(X_test)

        if cov_return_type is None:
            mu = self.gp.predict(X_test)
            var = None

            if self.normalize_y:
                mu = self._untransform_y(mu)

        else:
            predict_kwargs = {'return_cov': False, 'return_std': True}
            if cov_return_type == 'full_cov':
                predict_kwargs = {'return_cov': True, 'return_std': False}

            mu, var = self.gp.predict(X_test, **predict_kwargs)

            if cov_return_type != 'full_cov':
                var = var ** 2  # since we get standard deviation for faster computation

            # Clip negative variances and set them to the smallest
            # positive float value
            var = np.clip(var, VERY_SMALL_NUMBER, np.inf)

            if self.normalize_y:
                mu, var = self._untransform_y(mu, var)

            if cov_return_type == 'diagonal_std':
                var = np.sqrt(var)  # converting variance to std deviation if specified

        return mu, var

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        """
        Samples F function values from the current posterior at the N
        specified test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        n_funcs: int
            Number of function values that are drawn at each test point.

        Returns
        ----------
        function_samples: np.array(N, F)
            The F function values drawn at the N test points.
        """

        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        X_test = self._impute_inactive(X_test)
        funcs = self.gp.sample_y(X_test, n_samples=n_funcs, random_state=self.rng)

        if self.normalize_y:
            funcs = self._untransform_y(funcs)

        if len(funcs.shape) == 1:
            return funcs[None, :]
        else:
            return funcs
