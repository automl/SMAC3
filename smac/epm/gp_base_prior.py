import math

import numpy as np
import scipy.stats as sps

from smac.utils.constants import VERY_SMALL_NUMBER


class Prior(object):

    def __init__(self, rng: np.random.RandomState=None):
        """
        Abstract base class to define the interface for priors
        of GP hyperparameter.

        This class is a verbatim copy of the implementation of RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        rng: np.random.RandomState
            Random number generator

        """
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

    def lnprob(self, theta: np.ndarray):
        """
        Returns the log probability of theta. Note: theta should
        be on a log scale.

        Parameters
        ----------
        theta : (D,) numpy array
            A hyperparameter configuration in log space.

        Returns
        -------
        float
            The log probability of theta
        """
        pass

    def sample_from_prior(self, n_samples: int):
        """
        Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        (N, D) np.array
            The samples from the prior.
        """
        pass

    def gradient(self, theta: np.ndarray):
        """
        Computes the gradient of the prior with
        respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """
        pass


class TophatPrior(Prior):

    def __init__(self, l_bound: float, u_bound: float, rng: np.random.RandomState=None):
        """
        Tophat prior as it used in the original spearmint code.

        This class is a verbatim copy of the implementation of RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        l_bound : float
            Lower bound of the prior. Note the log scale.
        u_bound : float
            Upper bound of the prior. Note the log scale.
        rng: np.random.RandomState
            Random number generator
        """
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng
        self.min = l_bound
        self.max = u_bound
        if not (self.max > self.min):
            raise Exception("Upper bound of Tophat prior must be greater than the lower bound!")

    def lnprob(self, theta: np.ndarray):
        """
        Returns the log probability of theta. Note: theta should
        be on a log scale.

        Parameters
        ----------
        theta : (D,) numpy array
            A hyperparameter configuration in log space.

        Returns
        -------
        float
            The log probability of theta
        """
        if np.ndim(theta) == 0:
            if theta < self.min or theta > self.max:
                return -np.inf
            else:
                return 0
        else:
            if ((theta < self.min) | (theta > self.max)).any():
                return -np.inf
            else:
                return 0

    def sample_from_prior(self, n_samples: int):
        """
        Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        (N, D) np.array
            The samples from the prior.
        """

        p0 = self.min + self.rng.rand(n_samples) * (self.max - self.min)
        return p0[:, np.newaxis]

    def gradient(self, theta: np.ndarray):
        """
        Computes the gradient of the prior with
        respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array

            The gradient of the prior at theta.
        """
        if np.ndim(theta) == 0:
            return 0
        else:
            return np.zeros([theta.shape[0]])


class HorseshoePrior(Prior):

    def __init__(self, scale: float=0.1, rng: np.random.RandomState=None):
        """
        Horseshoe Prior as it is used in spearmint

        This class is a verbatim copy of the implementation of RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        scale: float
            Scaling parameter. See below how it is influenced
            the distribution.
        rng: np.random.RandomState
            Random number generator
        """
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng
        self.scale = scale
        self.scale_square = scale ** 2

    def lnprob(self, theta: np.ndarray):
        """
        Returns the log probability of theta. Note: theta should
        be on a log scale.

        Parameters
        ----------
        theta : (D,) numpy array
            A hyperparameter configuration in log space.

        Returns
        -------
        float
            The log probability of theta
        """
        # We computed it exactly as in the original spearmint code, they basically say that there's no analytical form
        # of the horseshoe prior, but that the multiplier is bounded between 2 and 4 and that they used the middle
        # See "The horseshoe estimator for sparse signals" by Carvalho, Poloson and Scott (2010), Equation 1.
        # https://www.jstor.org/stable/25734098
        # Compared to the paper by Carvalho, there's a constant multiplicator missing
        # Compared to Spearmint we first have to undo the log space transformation of the theta
        if np.ndim(theta) == 0:
            if theta == 0:
                return np.inf  # POSITIVE infinity (this is the "spike")
        else:
            if np.any(theta == 0.0):
                return np.inf  # POSITIVE infinity (this is the "spike")

        if np.ndim(theta) == 0:
            a = math.log(1 + 3.0 * (self.scale_square / math.exp(theta) ** 2))
            return math.log(a + VERY_SMALL_NUMBER)
        else:
            a = np.log(1 + 3.0 * (self.scale_square / np.exp(theta) ** 2))
            return np.log(sum(a) + VERY_SMALL_NUMBER)

    def sample_from_prior(self, n_samples: int):
        """
        Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        (N, D) np.array
            The samples from the prior.
        """

        lamda = np.abs(self.rng.standard_cauchy(size=n_samples))

        p0 = np.log(np.abs(self.rng.randn() * lamda * self.scale))
        return p0[:, np.newaxis]

    def gradient(self, theta: np.ndarray):
        """
        Computes the gradient of the prior with
        respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """
        if np.ndim(theta) == 0:
            if theta == 0:
                return np.inf  # POSITIVE infinity (this is the "spike")
            else:
                a = -(6 * self.scale_square)
                b = (3 * self.scale_square + math.exp(2 * theta))
                b *= np.log(3 * self.scale_square * math.exp(- 2 * theta) + 1)
                b = max(b, 1e-14)
                return a / b

        else:
            if np.any(theta == 0.0):
                return np.ones(theta.shape) * np.inf  # POSITIVE infinity (this is the "spike")
            else:
                a = -(6 * self.scale_square)
                b = (3 * self.scale_square + np.exp(2 * theta))
                b *= np.log(3 * self.scale_square * np.exp(- 2 * theta) + 1)
                b = np.maximum(b, 1e-14)
                return a / b


class LognormalPrior(Prior):
    def __init__(self, sigma: float, mean: float=0, rng: np.random.RandomState=None):
        """
        Log normal prior

        This class is a verbatim copy of the implementation of RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        sigma: float
            Specifies the standard deviation of the normal
            distribution.
        mean: float
            Specifies the mean of the normal distribution
        rng: np.random.RandomState
            Random number generator
        """
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        if mean != 0:
            raise NotImplementedError(mean)

        self.sigma = sigma
        self.sigma_square = sigma ** 2
        self.mean = mean
        self.sqrt_2_pi = np.sqrt(2 * np.pi)

    def lnprob(self, theta: np.ndarray):
        """
        Returns the log probability of theta. Note: theta should be on a log scale.

        Parameters
        ----------
        theta : (D,) numpy array
            A hyperparameter configuration in log space.

        Returns
        -------
        float
            The log probability of theta
        """
        if np.ndim(theta) == 0:
            if theta <= 0:
                return 0
            else:
                rval = (
                    -(math.log(theta) - self.mean) ** 2 / (2 * self.sigma_square)
                    - math.log(self.sqrt_2_pi * self.sigma * theta)
                )
                return rval

        rval = -(np.log(theta) - self.mean) ** 2 / (2 * self.sigma_square) - np.log(self.sqrt_2_pi * self.sigma * theta)
        # Alternative (slower) form of computation: sps.lognorm.logpdf(theta, self.sigma, loc=self.mean)
        if (~np.isfinite(rval)).any():
            rval[~np.isfinite(rval)] = 0
        return rval

    def sample_from_prior(self, n_samples: int):
        """
        Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        (N, D) np.array
            The samples from the prior.
        """

        p0 = self.rng.lognormal(mean=self.mean,
                                sigma=self.sigma,
                                size=n_samples)
        return p0[:, np.newaxis]

    def gradient(self, theta: np.ndarray):
        """
        Computes the gradient of the prior with
        respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """
        if np.ndim(theta) == 0:
            if theta <= 0:
                return 0
            else:
                return -(self.sigma_square + math.log(theta)) / (self.sigma_square * (theta))

        # derivative of log(1 / (x * s^2 * sqrt(2 pi)) * exp( - 0.5 * (log(x ) / s^2))^2))
        # This is without the mean!!!
        rval = - (self.sigma_square + np.log(theta)) / (self.sigma_square * (theta))
        if (~np.isfinite(rval)).any():
            rval[~np.isfinite(rval)] = 0
        return rval


class NormalPrior(Prior):
    def __init__(self, sigma: float, mean: float=0, rng: np.random.RandomState=None):
        """
        Normal prior

        This class is a verbatim copy of the implementation of RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        sigma: float
            Specifies the standard deviation of the normal
            distribution.
        mean: float
            Specifies the mean of the normal distribution
        rng: np.random.RandomState
            Random number generator
        """
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.sigma = sigma
        self.mean = mean

    def lnprob(self, theta: np.ndarray):
        """
        Returns the pdf of theta. Note: theta should
        be on a log scale.

        Parameters
        ----------
        theta : (D,) numpy array
            A hyperparameter configuration in log space.

        Returns
        -------
        float
            The log probability of theta
        """

        return sps.norm.pdf(theta, scale=self.sigma, loc=self.mean)

    def sample_from_prior(self, n_samples: int):
        """
        Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        (N, D) np.array
            The samples from the prior.
        """

        p0 = self.rng.normal(loc=self.mean,
                             scale=self.sigma,
                             size=n_samples)
        return p0[:, np.newaxis]

    def gradient(self, theta: np.ndarray):
        """
        Computes the gradient of the prior with
        respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) *\
               (- theta / (self.sigma ** 2) * np.exp(- (theta ** 2) / (2 * self.sigma ** 2)))


class LowerBoundPrior(Prior):
    def __init__(self, lower_bound=-20, rng: np.random.RandomState=None):
        super().__init__(rng=rng)
        self.lower_bound = lower_bound

    def lnprob(self, theta: np.ndarray):
        if np.ndim(theta) == 0:
            if theta < self.lower_bound:
                return - ((theta - self.lower_bound) ** 2)
            else:
                return 0
        else:
            raise NotImplementedError()

    def gradient(self, theta: np.ndarray):
        if np.ndim(theta) == 0:
            if theta < self.lower_bound:
                return -2 * (theta - self.lower_bound)
            else:
                return 0
        else:
            raise NotImplementedError()

