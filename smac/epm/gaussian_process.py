import logging
import typing

import george
import numpy as np
from scipy import optimize

from smac.epm import normalization
from smac.epm.base_gp import BaseModel
from smac.epm.gp_base_prior import BasePrior

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
        it implements the Prior interface.
    noise : float
        Noise term that is added to the diagonal of the covariance matrix
        for the Cholesky decomposition.
    use_gradients : bool
        Use gradient information to optimize the negative log likelihood
    normalize_output : bool
        Zero mean unit variance normalization of the output values
    normalize_input : bool
        Normalize all inputs to be in [0, 1]. This is important to define good priors for the
        length scales.
    rng: np.random.RandomState
        Random number generator
    """

    def __init__(
        self,
        types: np.ndarray,
        bounds: typing.List[typing.Tuple[float, float]],
        kernel: george.kernels.Kernel,
        prior: BasePrior=None,
        noise: float=1e-3,
        use_gradients: bool=False,
        normalize_output: bool=True,
        normalize_input: bool=True,
        rng: typing.Optional[np.random.RandomState]=None,
    ):

        super().__init__(types=types, bounds=bounds)

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.kernel = kernel
        self.gp = None
        self.prior = prior
        self.noise = noise
        self.use_gradients = use_gradients
        self.normalize_output = normalize_output
        self.normalize_input = normalize_input
        self.X = None
        self.y = None
        self.hypers = []
        self.is_trained = False

    def _train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool=True):
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

        # Use the empirical mean of the data as mean for the GP
        self.mean = np.mean(self.y, axis=0)
        self.gp = george.GP(self.kernel, mean=self.mean)

        if do_optimize:
            self.hypers = self._optimize()
            self.gp.kernel.set_parameter_vector(self.hypers[:-1])
            self.noise = np.exp(self.hypers[-1])  # sigma^2
        else:
            self.hypers = self.gp.kernel.get_parameter_vector()
            self.hypers = np.append(self.hypers, np.log(self.noise))

        try:
            self.gp.compute(self.X, yerr=np.sqrt(self.noise))
        except np.linalg.LinAlgError:
            self.noise *= 10
            self.gp.compute(self.X, yerr=np.sqrt(self.noise))

        self.is_trained = True

    def _get_noise(self):
        return self.noise

    def _nll(self, theta: np.ndarray) -> float:
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
        # Specify bounds to keep things sane
        if np.any((-20 > theta) + (theta > 20)):
            return 1e25

        # The last entry of theta is always the noise
        self.gp.kernel.set_parameter_vector(theta[:-1])
        noise = np.exp(theta[-1])  # sigma^2

        try:
            self.gp.compute(self.X, yerr=np.sqrt(noise))
        except np.linalg.LinAlgError:
            return 1e25

        ll = self.gp.lnlikelihood(self.y, quiet=True)

        # Add prior
        if self.prior is not None:
            ll += self.prior.lnprob(theta)

        # We add a minus here because scipy is minimizing
        return -ll if np.isfinite(ll) else 1e25

    def _grad_nll(self, theta: np.ndarray):

        self.gp.kernel.set_parameter_vector(theta[:-1])
        noise = np.exp(theta[-1])

        self.gp.compute(self.X, yerr=np.sqrt(noise))

        self.gp._compute_alpha(self.y)
        K_inv = self.gp.solver.apply_inverse(np.eye(self.gp._alpha.size),
                                             in_place=True)

        # The gradients of the Gram matrix, for the noise this is just
        # the identity matrix
        Kg = self.gp.kernel.gradient(self.gp._x)
        Kg = np.concatenate((Kg, np.eye(Kg.shape[0])[:, :, None]), axis=2)

        # Calculate the gradient.
        A = np.outer(self.gp._alpha, self.gp._alpha) - K_inv
        g = 0.5 * np.einsum('ijk,ij', Kg, A)

        if self.prior is not None:
            g += self.prior.gradient(theta)

        return -g

    def _optimize(self) -> np.ndarray:
        """
        Optimizes the marginal log likelihood and returns the best found
        hyperparameter configuration theta.

        Returns
        -------
        theta : np.ndarray(H)
            Hyperparameter vector that maximizes the marginal log likelihood
        """
        # Start optimization from the previous hyperparameter configuration
        p0 = self.gp.kernel.get_parameter_vector()
        p0 = np.append(p0, np.log(self.noise))

        if self.use_gradients:
            theta, _, _ = optimize.minimize(self._nll, p0,
                                            method="BFGS",
                                            jac=self._grad_nll)
        else:
            try:
                results = optimize.minimize(self._nll, p0, method='L-BFGS-B')
                theta = results.x
            except ValueError:
                logging.error("Could not find a valid hyperparameter configuration! Use initial configuration")
                theta = p0

        return theta

    def _predict(self, X_test: np.ndarray, full_cov: bool=False):
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        full_cov: bool
            If set to true than the whole covariance matrix between the test points is returned

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,) or np.array(N, N) if full_cov == True
            predictive variance

        """

        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        if self.normalize_input:
            X_test_norm, _, _ = normalization.zero_one_normalization(X_test, self.lower, self.upper)
        else:
            X_test_norm = X_test

        mu, var = self.gp.predict(self.y, X_test_norm)

        if self.normalize_output:
            mu = normalization.zero_mean_unit_var_unnormalization(mu, self.y_mean, self.y_std)
            var *= self.y_std ** 2
        if not full_cov:
            var = np.diag(var)

        # Clip negative variances and set them to the smallest
        # positive float value
        if var.shape[0] == 1:
            var = np.clip(var, np.finfo(var.dtype).eps, np.inf)
        else:
            var = np.clip(var, np.finfo(var.dtype).eps, np.inf)
            var[np.where((var < np.finfo(var.dtype).eps) & (var > -np.finfo(var.dtype).eps))] = 0

        return mu, var

    def sample_functions(self, X_test: np.ndarray, n_funcs: int=1) -> np.ndarray:
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
        function_samples: np.array(F, N)
            The F function values drawn at the N test points.
        """

        if self.normalize_input:
            X_test_norm, _, _ = normalization.zero_one_normalization(X_test, self.lower, self.upper)
        else:
            X_test_norm = X_test

        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        funcs = self.gp.sample_conditional(self.y, X_test_norm, n_funcs)

        if self.normalize_output:
            funcs = normalization.zero_mean_unit_var_unnormalization(funcs, self.y_mean, self.y_std)

        if len(funcs.shape) == 1:
            return funcs[None, :]
        else:
            return funcs
