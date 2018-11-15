from copy import deepcopy
import logging
import typing

import george
import emcee
import numpy as np

from smac.epm.base_gp import BaseModel
from smac.epm.gaussian_process import GaussianProcess
from smac.epm import normalization
from smac.epm.gp_base_prior import BasePrior

logger = logging.getLogger(__name__)


class GaussianProcessMCMC(BaseModel):

    def __init__(
        self,
        types: np.ndarray,
        bounds: typing.List[typing.Tuple[float, float]],
        kernel: george.kernels.Kernel,
        prior: BasePrior=None,
        n_hypers: int=20,
        chain_length: int=2000,
        burnin_steps: int=2000,
        normalize_output: bool=True,
        normalize_input: bool=True,
        rng: typing.Optional[np.random.RandomState]=None,
        noise: int=-8,
    ):
        """
        Gaussian process model.

        The GP hyperparameters are integrated out by MCMC. If you use this class
        make sure that you also use an integrated acquisition function to
        integrate over the GP's hyperparameter as proposed by Snoek et al.

        This code is based on the implementation of RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        types : np.ndarray (D)
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass np.array([2, 0]). Note that we count starting from 0.
        bounds : list
            Specifies the bounds for continuous features.
        kernel : george kernel object
            Specifies the kernel that is used for all Gaussian Process
        prior : prior object
            Defines a prior for the hyperparameters of the GP. Make sure that
            it implements the Prior interface. During MCMC sampling the
            lnlikelihood is multiplied with the prior.
        n_hypers : int
            The number of hyperparameter samples. This also determines the
            number of walker for MCMC sampling as each walker will
            return one hyperparameter sample.
        chain_length : int
            The length of the MCMC chain. We start n_hypers walker for
            chain_length steps and we use the last sample
            in the chain as a hyperparameter sample.
        burnin_steps : int
            The number of burnin steps before the actual MCMC sampling starts.
        normalize_output : bool
            Zero mean unit variance normalization of the output values
        normalize_input : bool
            Normalize all inputs to be in [0, 1]. This is important to define good priors for the
            length scales.
        rng: np.random.RandomState
            Random number generator
        noise : float
            Noise term that is added to the diagonal of the covariance matrix
            for the Cholesky decomposition.
        """
        super().__init__(types=types, bounds=bounds)

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.kernel = kernel
        self.prior = prior
        self.noise = noise
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        self.models = []
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input
        self.X = None
        self.y = None
        self.is_trained = False

    def _train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool=True):
        """
        Performs MCMC sampling to sample hyperparameter configurations from the
        likelihood and trains for each sample a GP on X and y

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true we perform MCMC sampling otherwise we just use the
            hyperparameter specified in the kernel.
        """

        if self.normalize_input:
            # Normalize input to be in [0, 1]
            self.X, self.lower, self.upper = normalization.zero_one_normalization(X, self.lower, self.upper)
        else:
            self.X = X

        if len(y.shape) > 1:
            y = y.flatten()
            if len(y) != len(X):
                raise ValueError('Shape mismatch: %s vs %s' % (y.shape, X.shape))

        if self.normalize_output:
            # Normalize output to have zero mean and unit standard deviation
            self.y, self.y_mean, self.y_std = normalization.zero_mean_unit_var_normalization(y)
            if self.y_std == 0:
                raise ValueError("Cannot normalize output. All targets have the same value")
        else:
            self.y = y

        # Use the mean of the data as mean for the GP
        self.mean = np.mean(self.y, axis=0)
        self.gp = george.GP(self.kernel, mean=self.mean)

        if do_optimize:
            # We have one walker for each hyperparameter configuration
            sampler = emcee.EnsembleSampler(self.n_hypers,
                                            len(self.kernel) + 1,
                                            self._loglikelihood)
            sampler.random_state = self.rng.get_state()
            # Do a burn-in in the first iteration
            if not self.burned:
                # Initialize the walkers by sampling from the prior
                if self.prior is None:
                    self.p0 = self.rng.rand(self.n_hypers, len(self.kernel) + 1)
                else:
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                # Run MCMC sampling
                self.p0, _, _ = sampler.run_mcmc(self.p0,
                                                 self.burnin_steps,
                                                 rstate0=self.rng)

                self.burned = True

            # Start sampling
            pos, _, _ = sampler.run_mcmc(self.p0,
                                         self.chain_length,
                                         rstate0=self.rng)

            # Save the current position, it will be the start point in
            # the next iteration
            self.p0 = pos

            # Take the last samples from each walker
            self.hypers = sampler.chain[:, -1]

        else:
            self.hypers = self.gp.kernel.get_parameter_vector().tolist()
            self.hypers.append(self.noise)
            self.hypers = [self.hypers]

        self.models = []
        for sample in self.hypers:

            # Instantiate a GP for each hyperparameter configuration
            kernel = deepcopy(self.kernel)
            kernel.set_parameter_vector(sample[:-1])
            noise = np.exp(sample[-1])
            model = GaussianProcess(
                types=self.types,
                bounds=self.bounds,
                kernel=kernel,
                normalize_output=self.normalize_output,
                normalize_input=self.normalize_input,
                noise=noise,
                rng=self.rng,
            )
            model._train(X, y, do_optimize=False)
            self.models.append(model)

        self.is_trained = True

    def _loglikelihood(self, theta: np.ndarray):
        """
        Return the loglikelihood (+ the prior) for a hyperparameter
        configuration theta.

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

        # Bound the hyperparameter space to keep things sane. Note all
        # hyperparameters live on a log scale
        if np.any((-20 > theta) + (theta > 20)):
            return -np.inf

        # The last entry is always the noise
        sigma_2 = np.exp(theta[-1])
        # Update the kernel and compute the lnlikelihood.
        self.gp.kernel.set_parameter_vector(theta[:-1])

        try:
            self.gp.compute(self.X, yerr=np.sqrt(sigma_2))
        except:
            return -np.inf

        if self.prior is not None:
            return self.prior.lnprob(theta) + self.gp.lnlikelihood(self.y, quiet=True)
        else:
            return self.gp.lnlikelihood(self.y, quiet=True)

    def _predict(self, X_test: np.ndarray):
        r"""
        Returns the predictive mean and variance of the objective function
        at X average over all hyperparameter samples.
        The mean is computed by:
        :math \mu(x) = \frac{1}{M}\sum_{i=1}^{M}\mu_m(x)
        And the variance by:
        :math \sigma^2(x) = (\frac{1}{M}\sum_{i=1}^{M}(\sigma^2_m(x) + \mu_m(x)^2) - \mu^2

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """
        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        mu = np.zeros([len(self.models), X_test.shape[0]])
        var = np.zeros([len(self.models), X_test.shape[0]])
        for i, model in enumerate(self.models):
            mu_tmp, var_tmp = model.predict(X_test)
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
