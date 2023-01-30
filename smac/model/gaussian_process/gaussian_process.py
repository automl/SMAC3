from __future__ import annotations

from typing import Any, Optional, TypeVar, cast

import logging

import numpy as np
from ConfigSpace import ConfigurationSpace
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from smac.constants import VERY_SMALL_NUMBER
from smac.model.gaussian_process.abstract_gaussian_process import (
    AbstractGaussianProcess,
)
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = logging.getLogger(__name__)


Self = TypeVar("Self", bound="GaussianProcess")


class GaussianProcess(AbstractGaussianProcess):
    """Implementation of Gaussian process model. The Gaussian process hyperparameters are obtained by optimizing
    the marginal log likelihood.

    This code is based on the implementation of RoBO:
    Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
    RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
    In: NIPS 2017 Bayesian Optimization Workshop

    Parameters
    ----------
    configspace : ConfigurationSpace
    kernel : Kernel
        Kernel which is used for the Gaussian process.
    n_restarts : int, defaults to 10
        Number of restarts for the Gaussian process hyperparameter optimization.
    normalize_y : bool, defaults to True
        Zero mean unit variance normalization of the output values.
    instance_features : dict[str, list[int | float]] | None, defaults to None
        Features (list of int or floats) of the instances (str). The features are incorporated into the X data,
        on which the model is trained on.
    pca_components : float, defaults to 7
        Number of components to keep when using PCA to reduce dimensionality of instance features.
    seed : int
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        kernel: Kernel,
        n_restarts: int = 10,
        normalize_y: bool = True,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
    ):
        super().__init__(
            configspace=configspace,
            seed=seed,
            kernel=kernel,
            instance_features=instance_features,
            pca_components=pca_components,
        )

        self._normalize_y = normalize_y
        self._n_restarts = n_restarts

        # Internal variables
        self._hypers = np.empty((0,))
        self._is_trained = False
        self._n_ll_evals = 0

        self._set_has_conditions()

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"n_restarts": self._n_restarts, "normalize_y": self._normalize_y})

        return meta

    def _train(
        self: Self,
        X: np.ndarray,
        y: np.ndarray,
        optimize_hyperparameters: bool = True,
    ) -> Self:
        """Computes the Cholesky decomposition of the covariance of X and estimates the GP
        hyperparameters by optimizing the marginal log likelihood. The prior mean of the GP is set to
        the empirical mean of X.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameters + #features]
            Input data points.
        Y : np.ndarray [#samples, #objectives]
            The corresponding target values.
        optimize_hyperparameters: boolean
            If set to true, the hyperparameters are optimized, otherwise the default hyperparameters of the kernel are
            used.
        """
        if self._normalize_y:
            y = self._normalize(y)

        X = self._impute_inactive(X)
        y = y.flatten()

        n_tries = 10
        for i in range(n_tries):
            try:
                self._gp = self._get_gaussian_process()
                self._gp.fit(X, y)
                break
            except np.linalg.LinAlgError as e:
                if i == n_tries:
                    raise e

                # Assume that the last entry of theta is the noise
                theta = np.exp(self._kernel.theta)
                theta[-1] += 1
                self._kernel.theta = np.log(theta)

        if optimize_hyperparameters:
            self._all_priors = self._get_all_priors(add_bound_priors=False)
            self._hypers = self._optimize()
            self._gp.kernel.theta = self._hypers
            self._gp.fit(X, y)
        else:
            self._hypers = self._gp.kernel.theta

        # Set the flag
        self._is_trained = True

        return self

    def _get_gaussian_process(self) -> GaussianProcessRegressor:
        return GaussianProcessRegressor(
            kernel=self._kernel,
            normalize_y=False,  # We do not use scikit-learn's normalize routine
            optimizer=None,
            n_restarts_optimizer=0,  # We do not use scikit-learn's optimization routine
            alpha=0,  # Governed by the kernel
            random_state=self._rng,
        )

    def _nll(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Returns the negative marginal log likelihood (+ the prior) for a hyperparameter
        configuration theta. Negative because we use scipy minimize for optimization.

        Parameters
        ----------
        theta : np.ndarray
            Hyperparameter vector. Note that all hyperparameter are on a log scale.
        """
        self._n_ll_evals += 1

        try:
            lml, grad = self._gp.log_marginal_likelihood(theta, eval_gradient=True)
        except np.linalg.LinAlgError:
            return 1e25, np.zeros(theta.shape)

        for dim, priors in enumerate(self._all_priors):
            for prior in priors:
                lml += prior.get_log_probability(theta[dim])
                grad[dim] += prior.get_gradient(theta[dim])

        # We add a minus here because scipy is minimizing
        if not np.isfinite(lml).all() or not np.all(np.isfinite(grad)):
            return 1e25, np.zeros(theta.shape)
        else:
            return -lml, -grad

    def _optimize(self) -> np.ndarray:
        """Optimizes the marginal log likelihood and returns the best found hyperparameter
        configuration theta.

        Returns
        -------
        theta : np.ndarray
            Hyperparameter vector that maximizes the marginal log likelihood.
        """
        log_bounds = [(b[0], b[1]) for b in self._gp.kernel.bounds]

        # Start optimization from the previous hyperparameter configuration
        p0 = [self._gp.kernel.theta]
        if self._n_restarts > 0:
            dim_samples = []

            prior: list[AbstractPrior] | AbstractPrior | None = None
            for dim, hp_bound in enumerate(log_bounds):
                prior = self._all_priors[dim]
                # Always sample from the first prior
                if isinstance(prior, list):
                    if len(prior) == 0:
                        prior = None
                    else:
                        prior = prior[0]

                prior = cast(Optional[AbstractPrior], prior)
                if prior is None:
                    try:
                        sample = self._rng.uniform(
                            low=hp_bound[0],
                            high=hp_bound[1],
                            size=(self._n_restarts,),
                        )
                    except OverflowError:
                        raise ValueError("OverflowError while sampling from (%f, %f)" % (hp_bound[0], hp_bound[1]))

                    dim_samples.append(sample.flatten())
                else:
                    dim_samples.append(prior.sample_from_prior(self._n_restarts).flatten())
            p0 += list(np.vstack(dim_samples).transpose())

        theta_star: np.ndarray | None = None
        f_opt_star = np.inf
        for i, start_point in enumerate(p0):
            theta, f_opt, _ = optimize.fmin_l_bfgs_b(self._nll, start_point, bounds=log_bounds)
            if f_opt < f_opt_star:
                f_opt_star = f_opt
                theta_star = theta

        if theta_star is None:
            raise RuntimeError

        return theta_star

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not self._is_trained:
            raise Exception("Model has to be trained first!")

        X_test = self._impute_inactive(X)

        if covariance_type is None:
            mu = self._gp.predict(X_test)
            var = None

            if self._normalize_y:
                mu = self._untransform_y(mu)
        else:
            predict_kwargs = {"return_cov": False, "return_std": True}
            if covariance_type == "full":
                predict_kwargs = {"return_cov": True, "return_std": False}

            mu, var = self._gp.predict(X_test, **predict_kwargs)

            if covariance_type != "full":
                var = var**2  # Since we get standard deviation for faster computation

            # Clip negative variances and set them to the smallest
            # positive float value
            var = np.clip(var, VERY_SMALL_NUMBER, np.inf)

            if self._normalize_y:
                mu, var = self._untransform_y(mu, var)

            if covariance_type == "std":
                var = np.sqrt(var)  # Converting variance to std deviation if specified

        return mu, var

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        """Samples F function values from the current posterior at the N specified test points.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameters + #features]
            Input data points.
        n_funcs: int
            Number of function values that are drawn at each test point.

        Returns
        -------
        function_samples : np.ndarray
            The F function values drawn at the N test points.
        """
        if not self._is_trained:
            raise Exception("Model has to be trained first.")

        X_test = self._impute_inactive(X_test)
        funcs = self._gp.sample_y(X_test, n_samples=n_funcs, random_state=self._rng)

        if self._normalize_y:
            funcs = self._untransform_y(funcs)

        if len(funcs.shape) == 1:
            return funcs[None, :]
        else:
            return funcs
