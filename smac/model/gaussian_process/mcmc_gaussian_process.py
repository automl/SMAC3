from __future__ import annotations

from typing import Any, Optional, TypeVar, cast

import logging
import warnings
from copy import deepcopy

import emcee
import numpy as np
from ConfigSpace import ConfigurationSpace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from smac.model.gaussian_process.abstract_gaussian_process import (
    AbstractGaussianProcess,
)
from smac.model.gaussian_process.gaussian_process import GaussianProcess
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = logging.getLogger(__name__)
Self = TypeVar("Self", bound="MCMCGaussianProcess")


class MCMCGaussianProcess(AbstractGaussianProcess):
    """Implementation of a Gaussian process model which out-integrates its hyperparameters by
    Markow-Chain-Monte-Carlo (MCMC). If you use this class make sure that you also use an integrated acquisition
    function to integrate over the GP's hyperparameter as proposed by Snoek et al.

    This code is based on the implementation of RoBO:

    Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
    RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
    In: NIPS 2017 Bayesian Optimization Workshop

    Parameters
    ----------
    configspace : ConfigurationSpace
    kernel : Kernel
        Kernel which is used for the Gaussian process.
    n_mcmc_walkers : int, defaults to 20
        The number of hyperparameter samples. This also determines the number of walker for MCMC sampling as each
        walker will return one hyperparameter sample.
    chain_length : int, defaults to 50
        The length of the MCMC chain. We start `n_mcmc_walkers` walker for `chain_length` steps, and we use the last
        sample in the chain as a hyperparameter sample.
    burning_steps : int, defaults to 50
        The number of burning steps before the actual MCMC sampling starts.
    mcmc_sampler : str, defaults to "emcee"
        Choose a self-tuning MCMC sampler. Can be either ``emcee`` or ``nuts``.
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
        n_mcmc_walkers: int = 20,
        chain_length: int = 50,
        burning_steps: int = 50,
        mcmc_sampler: str = "emcee",
        average_samples: bool = False,
        normalize_y: bool = True,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
    ):
        if mcmc_sampler not in ["emcee", "nuts"]:
            raise ValueError(f"MCMC Gaussian process does not support the sampler `{mcmc_sampler}`.")

        super().__init__(
            configspace=configspace,
            kernel=kernel,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )

        self._n_mcmc_walkers = n_mcmc_walkers
        self._chain_length = chain_length
        self._burning_steps = burning_steps
        self._models: list[GaussianProcess] = []
        self._normalize_y = normalize_y
        self._mcmc_sampler = mcmc_sampler
        self._average_samples = average_samples
        self._set_has_conditions()

        # Internal statistics
        self._n_ll_evals = 0
        self._burned = False
        self._is_trained = False
        self._samples: np.ndarray | None = None

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "n_mcmc_walkers": self._n_mcmc_walkers,
                "chain_length": self._chain_length,
                "burning_steps": self._burning_steps,
                "mcmc_sampler": self._mcmc_sampler,
                "average_samples": self._average_samples,
                "normalize_y": self._normalize_y,
            }
        )

        return meta

    @property
    def models(self) -> list[GaussianProcess]:
        """Returns the internally used gaussian processes."""
        return self._models

    def _train(
        self: Self,
        X: np.ndarray,
        y: np.ndarray,
        optimize_hyperparameters: bool = True,
    ) -> Self:
        """Performs MCMC sampling to sample hyperparameter configurations from the likelihood and
        trains for each sample a Gaussian process on X and y.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameters + #features]
            Input data points.
        Y : np.ndarray [#samples, #objectives]
            The corresponding target values.
        optimize_hyperparameters: boolean
            If set to true, we perform MCMC sampling. Otherwise, we just use the hyperparameter specified in the kernel.
        """
        X = self._impute_inactive(X)
        if self._normalize_y:
            # A note on normalization for the Gaussian process with MCMC:
            # Scikit-learn uses a different "normalization" than we use in SMAC3. Scikit-learn normalizes the data to
            # have zero mean, while we normalize it to have zero mean unit variance. To make sure the scikit-learn GP
            # behaves the same when we use it directly or indirectly (through the gaussian_process.py file), we
            # normalize the data here. Then, after the individual GPs are fit, we inject the statistics into them, so
            # they unnormalize the data at prediction time.
            y = self._normalize(y)

        self._gp = self._get_gaussian_process()

        if optimize_hyperparameters:
            self._gp.fit(X, y)
            self._all_priors = self._get_all_priors(
                add_bound_priors=True,
                add_soft_bounds=True if self._mcmc_sampler == "nuts" else False,
            )

            if self._mcmc_sampler == "emcee":
                sampler = emcee.EnsembleSampler(self._n_mcmc_walkers, len(self._kernel.theta), self._ll)
                sampler.random_state = self._rng.get_state()
                # Do a burn-in in the first iteration
                if not self._burned:
                    # Initialize the walkers by sampling from the prior
                    dim_samples = []

                    prior: AbstractPrior | list[AbstractPrior] | None = None
                    for dim, prior in enumerate(self._all_priors):
                        # Always sample from the first prior
                        if isinstance(prior, list):
                            if len(prior) == 0:
                                prior = None
                            else:
                                prior = prior[0]
                        prior = cast(Optional[AbstractPrior], prior)
                        if prior is None:
                            raise NotImplementedError()
                        else:
                            dim_samples.append(prior.sample_from_prior(self._n_mcmc_walkers).flatten())
                    self.p0 = np.vstack(dim_samples).transpose()

                    # Run MCMC sampling
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", r"invalid value encountered in double_scalars.*")
                        self.p0, _, _ = sampler.run_mcmc(self.p0, self._burning_steps)

                    self.burned = True

                # Start sampling & save the current position, it will be the start point in the next iteration
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", r"invalid value encountered in double_scalars.*")
                    self.p0, _, _ = sampler.run_mcmc(self.p0, self._chain_length)

                # Take the last samples from each walker
                self._samples = sampler.get_chain()[-1]
            elif self._mcmc_sampler == "nuts":
                # Originally published as:
                # http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf
                # A good explanation of HMC:
                # https://theclevermachine.wordpress.com/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/
                # A good explanation of HMC and NUTS can be found in:
                # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12681

                # Do not require the installation of NUTS for SMAC
                # This requires NUTS from https://github.com/mfeurer/NUTS
                import nuts.nuts  # type: ignore

                # Perform initial fit to the data to obtain theta0
                if not self.burned:
                    theta0 = self._gp.kernel.theta
                    self._burned = True
                else:
                    theta0 = self.p0
                samples, _, _ = nuts.nuts.nuts6(
                    f=self._ll_w_grad,
                    Madapt=self._burning_steps,
                    M=self._chain_length,
                    theta0=theta0,
                    # Increasing this value results in longer running times
                    delta=0.5,
                    adapt_mass=False,
                    # Rather low max depth to keep the number of required gradient steps low
                    max_depth=10,
                    rng=self._rng,
                )
                indices = [int(np.rint(ind)) for ind in np.linspace(start=0, stop=len(samples) - 1, num=10)]
                self._samples = samples[indices]

                assert self._samples is not None
                self.p0 = self._samples.mean(axis=0)
            else:
                raise ValueError(self._mcmc_sampler)

            if self._average_samples:
                assert self._samples is not None
                self._samples = [self._samples.mean(axis=0)]  # type: ignore

        else:
            self._samples = self._gp.kernel.theta
            self._samples = [self._samples]  # type: ignore

        self._models = []

        assert self._samples is not None
        for sample in self._samples:

            if (sample < -50).any():
                sample[sample < -50] = -50
            if (sample > 50).any():
                sample[sample > 50] = 50

            # Instantiate a GP for each hyperparameter configuration
            kernel = deepcopy(self._kernel)
            kernel.theta = sample
            model = GaussianProcess(
                configspace=self._configspace,
                kernel=kernel,
                normalize_y=False,
                seed=self._rng.randint(low=0, high=10000),
            )
            try:
                model._train(X, y, optimize_hyperparameters=False)
                self._models.append(model)
            except np.linalg.LinAlgError:
                pass

        if len(self._models) == 0:
            kernel = deepcopy(self._kernel)
            kernel.theta = self.p0
            model = GaussianProcess(
                configspace=self._configspace,
                kernel=kernel,
                normalize_y=False,
                seed=self._rng.randint(low=0, high=10000),
            )
            model._train(X, y, optimize_hyperparameters=False)
            self._models.append(model)

        if self._normalize_y:
            # Inject the normalization statistics into the individual models. Setting normalize_y to True makes the
            # individual GPs unnormalize the data at predict time.
            for model in self._models:
                model._normalize_y = True
                model.mean_y_ = self.mean_y_
                model.std_y_ = self.std_y_

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

    def _ll(self, theta: np.ndarray) -> float:
        """Returns the marginal log likelihood (+ the prior) for a hyperparameter configuration
        theta.

        Parameters
        ----------
        theta : np.ndarray
            Hyperparameter vector. Note that all hyperparameters are on a log scale.
        """
        self._n_ll_evals += 1

        # Bound the hyperparameter space to keep things sane. Note that all
        # hyperparameters live on a log scale.
        if (theta < -50).any():
            theta[theta < -50] = -50
        if (theta > 50).any():
            theta[theta > 50] = 50

        try:
            lml = self._gp.log_marginal_likelihood(theta)
        except ValueError:
            return -np.inf

        # Add prior
        for dim, priors in enumerate(self._all_priors):
            for prior in priors:
                lml += prior.get_log_probability(theta[dim])

        if not np.isfinite(lml):
            return -np.inf
        else:
            return lml

    def _ll_w_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Returns the marginal log likelihood (+ the prior) for a hyperparameter configuration
        theta.

        Parameters
        ----------
        theta : np.ndarray
            Hyperparameter vector. Note that all hyperparameter are on a log scale.
        """
        self._n_ll_evals += 1

        # Bound the hyperparameter space to keep things sane. Note that all hyperparameters live on a log scale.
        if (theta < -50).any():
            theta[theta < -50] = -50
        if (theta > 50).any():
            theta[theta > 50] = 50

        lml = 0.0
        grad = np.zeros(theta.shape)

        # Add prior
        for dim, priors in enumerate(self._all_priors):
            for prior in priors:
                lml += prior.get_log_probability(theta[dim])
                grad[dim] += prior.get_gradient(theta[dim])

        # Check if one of the priors is invalid, if so, no need to compute the log marginal likelihood
        if lml < -1e24:
            return -1e25, np.zeros(theta.shape)

        try:
            lml_, grad_ = self._gp.log_marginal_likelihood(theta, eval_gradient=True)
            lml += lml_
            grad += grad_
        except ValueError:
            return -1e25, np.zeros(theta.shape)

        # We add a minus here because scipy is minimizing
        if not np.isfinite(lml) or (~np.isfinite(grad)).any():
            return -1e25, np.zeros(theta.shape)
        else:
            return lml, grad

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        r"""
        Returns the predictive mean and variance of the objective function
        at X averaged over all hyperparameter samples.

        The mean is computed by:
        :math \mu(x) = \frac{1}{M}\sum_{i=1}^{M}\mu_m(x)

        And the variance by:
        :math \sigma^2(x) = (\frac{1}{M}\sum_{i=1}^{M}(\sigma^2_m(x) + \mu_m(x)^2) - \mu^2
        """
        if not self._is_trained:
            raise Exception("Model has to be trained first!")

        if covariance_type != "diagonal":
            raise ValueError("`covariance_type` can only take `diagonal` for this model.")

        X_test = self._impute_inactive(X)

        mu = np.zeros([len(self._models), X_test.shape[0]])
        var = np.zeros([len(self._models), X_test.shape[0]])
        for i, model in enumerate(self._models):
            mu_tmp, var_tmp = model.predict(X_test)
            assert var_tmp is not None
            mu[i] = mu_tmp.flatten()
            var[i] = var_tmp.flatten()

        m = mu.mean(axis=0)

        # See the Algorithm Runtime Prediction paper by Hutter et al.
        # for the derivation of the total variance
        v = np.var(mu, axis=0) + np.mean(var, axis=0)

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        return m, v
