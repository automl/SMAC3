import math
import warnings

import numpy as np
import scipy.stats as sps

from smac.utils.constants import VERY_SMALL_NUMBER

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class Prior(object):
    """Abstract base class to define the interface for priors of GP hyperparameter.

    This class is adapted from RoBO:

    Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
    RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
    In: NIPS 2017 Bayesian Optimization Workshop

    [16.04.2019]: Whenever lnprob or the gradient is computed for a scalar input, we use math.* rather than np.*

    Parameters
    ----------
    rng: np.random.RandomState
        Random number generator
    """

    def __init__(self, rng: np.random.RandomState):
        if rng is None:
            raise ValueError("Argument rng must not be `None`.")
        self.rng = rng

    def lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Theta must be on a log scale! This method exponentiates theta and calls ``self._lnprob``.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in log space.

        Returns
        -------
        float
            The log probability of theta
        """
        return self._lnprob(np.exp(theta))

    def _lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Theta must be on the original scale.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration on the original scale.

        Returns
        -------
        float
            The log probability of theta
        """
        raise NotImplementedError()

    def sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns ``n_samples`` from the prior.

        All samples are on a log scale. This method calls ``self._sample_from_prior`` and applies a log transformation
        to the obtained values.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        if np.ndim(n_samples) != 0:
            raise ValueError("argument n_samples needs to be a scalar (is %s)" % n_samples)
        if n_samples <= 0:
            raise ValueError("argument n_samples needs to be positive (is %d)" % n_samples)

        sample = np.log(self._sample_from_prior(n_samples=n_samples))

        if np.any(~np.isfinite(sample)):
            raise ValueError("Sample %s from prior %s contains infinite values!" % (sample, self))

        return sample

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns ``n_samples`` from the prior.

        All samples are on a original scale.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        raise NotImplementedError()

    def gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Theta must be on the original scale.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in log space

        Returns
        -------
        float
            The gradient of the prior at theta.
        """
        return self._gradient(np.exp(theta))

    def _gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in the original space space

        Returns
        -------
        float
            The gradient of the prior at theta.
        """
        raise NotImplementedError()


class TophatPrior(Prior):
    def __init__(self, lower_bound: float, upper_bound: float, rng: np.random.RandomState):
        """Tophat prior as it used in the original spearmint code.

        This class is adapted from RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        lower_bound : float
            Lower bound of the prior. In original scale.
        upper_bound : float
            Upper bound of the prior. In original scale.
        rng: np.random.RandomState
            Random number generator
        """
        super().__init__(rng)
        self.min = lower_bound
        self._log_min = np.log(lower_bound)
        self.max = upper_bound
        self._log_max = np.log(upper_bound)
        if not (self.max > self.min):
            raise Exception("Upper bound of Tophat prior must be greater than the lower bound!")

    def _lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Parameters
        ----------
        theta : float
            A hyperparameter configuration

        Returns
        -------
        float
        """
        if theta < self.min or theta > self.max:
            return -np.inf
        else:
            return 0

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Return ``n_samples`` from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        if np.ndim(n_samples) != 0:
            raise ValueError("argument n_samples needs to be a scalar (is %s)" % n_samples)
        if n_samples <= 0:
            raise ValueError("argument n_samples needs to be positive (is %d)" % n_samples)

        p0 = np.exp(self.rng.uniform(low=self._log_min, high=self._log_max, size=(n_samples,)))
        return p0

    def gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array

            The gradient of the prior at theta.
        """
        return 0


class HorseshoePrior(Prior):
    def __init__(self, scale: float, rng: np.random.RandomState):
        """Horseshoe Prior as it is used in spearmint.

        This class is adapted from RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        scale: float
            Scaling parameter. See below how it is influencing the distribution.
        rng: np.random.RandomState
            Random number generator
        """
        super().__init__(rng)
        self.scale = scale
        self.scale_square = scale**2

    def _lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Parameters
        ----------
        theta : (D,) numpy array
            A hyperparameter configuration

        Returns
        -------
        float
        """
        # We computed it exactly as in the original spearmint code, they basically say that there's no analytical form
        # of the horseshoe prior, but that the multiplier is bounded between 2 and 4 and that they used the middle
        # See "The horseshoe estimator for sparse signals" by Carvalho, Poloson and Scott (2010), Equation 1.
        # https://www.jstor.org/stable/25734098
        # Compared to the paper by Carvalho, there's a constant multiplicator missing
        # Compared to Spearmint we first have to undo the log space transformation of the theta
        # Note: "undo log space transformation" is done in parent class
        if theta == 0:
            return np.inf  # POSITIVE infinity (this is the "spike")
        else:
            a = math.log(1 + 3.0 * (self.scale_square / theta**2))
            return math.log(a + VERY_SMALL_NUMBER)

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        # This is copied from RoBO - scale is most likely the tau parameter
        lamda = np.abs(self.rng.standard_cauchy(size=n_samples))
        p0 = np.abs(self.rng.randn() * lamda * self.scale)
        return p0

    def _gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """
        if theta == 0:
            return np.inf  # POSITIVE infinity (this is the "spike")
        else:
            a = -(6 * self.scale_square)
            b = 3 * self.scale_square + theta**2
            b *= math.log(3 * self.scale_square * theta ** (-2) + 1)
            b = max(b, 1e-14)
            return a / b


class LognormalPrior(Prior):
    def __init__(
        self,
        sigma: float,
        rng: np.random.RandomState,
        mean: float = 0,
    ):
        """Log normal prior.

        This class is adapted from RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        sigma: float
            Specifies the standard deviation of the normal
            distribution.
        rng: np.random.RandomState
            Random number generator
        mean: float
            Specifies the mean of the normal distribution
        """
        super().__init__(rng)

        if mean != 0:
            raise NotImplementedError(mean)

        self.sigma = sigma
        self.sigma_square = sigma**2
        self.mean = mean
        self.sqrt_2_pi = np.sqrt(2 * np.pi)

    def _lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Parameters
        ----------
        theta : float
            A hyperparameter configuration

        Returns
        -------
        float
        """
        if theta <= self.mean:
            return -1e25
        else:
            rval = -((math.log(theta) - self.mean) ** 2) / (2 * self.sigma_square) - math.log(
                self.sqrt_2_pi * self.sigma * theta
            )
            return rval

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        return self.rng.lognormal(mean=self.mean, sigma=self.sigma, size=n_samples)

    def _gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """
        if theta <= 0:
            return 0
        else:
            # derivative of log(1 / (x * s^2 * sqrt(2 pi)) * exp( - 0.5 * (log(x ) / s^2))^2))
            # This is without the mean!!!
            return -(self.sigma_square + math.log(theta)) / (self.sigma_square * (theta)) * theta


class SoftTopHatPrior(Prior):
    def __init__(self, lower_bound: float, upper_bound: float, exponent: float, rng: np.random.RandomState) -> None:
        super().__init__(rng)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.lower_bound = lower_bound
            try:
                self._log_lower_bound = np.log(lower_bound)
            except RuntimeWarning as w:
                if "invalid value encountered in log" in w.args[0]:
                    raise ValueError("Invalid lower bound %f (cannot compute log)" % lower_bound)

                raise w
            self.upper_bound = upper_bound
            try:
                self._log_upper_bound = np.log(upper_bound)
            except RuntimeWarning as w:
                if "invalid value encountered in log" in w.args[0]:
                    raise ValueError("Invalid lower bound %f (cannot compute log)" % lower_bound)

                raise w

        if exponent <= 0:
            raise ValueError("Exponent cannot be less or equal than zero (but is %f)" % exponent)
        self.exponent = exponent

    def lnprob(self, theta: float) -> float:
        """Return the log probability of theta."""
        # We need to use lnprob here instead of _lnprob to have the squared function work
        # in the logarithmic space, too.
        if np.ndim(theta) == 0:
            if theta < self._log_lower_bound:
                return -((theta - self._log_lower_bound) ** self.exponent)
            elif theta > self._log_upper_bound:
                return -((self._log_upper_bound - theta) ** self.exponent)
            else:
                return 0
        else:
            raise NotImplementedError()

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        return np.exp(self.rng.uniform(self._log_lower_bound, self._log_upper_bound, size=(n_samples,)))

    def gradient(self, theta: float) -> float:
        """Returns the gradient of the prior at theta."""
        if np.ndim(theta) == 0:
            if theta < self._log_lower_bound:
                return -self.exponent * (theta - self._log_lower_bound)
            elif theta > self._log_upper_bound:
                return self.exponent * (self._log_upper_bound - theta)
            else:
                return 0
        else:
            raise NotImplementedError()

    def __repr__(self) -> str:
        return "SoftTopHatPrior(lower_bound=%f, upper_bound=%f)" % (
            self.lower_bound,
            self.upper_bound,
        )


class GammaPrior(Prior):
    def __init__(self, a: float, scale: float, loc: float, rng: np.random.RandomState):
        """Gamma prior.

        f(x) = (x-loc)**(a-1) * e**(-(x-loc)) * (1/scale)**a / gamma(a)

        Parameters
        ----------
        a: float > 0
            shape parameter
        scale: float > 0
            scale parameter (1/scale corresponds to parameter p in canonical form)
        loc: float
            mean parameter for the distribution
        rng: np.random.RandomState
            Random number generator
        """
        super().__init__(rng)

        self.a = a
        self.loc = loc
        self.scale = scale

    def _lnprob(self, theta: float) -> float:
        """Returns the logpdf of theta.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration

        Returns
        -------
        float
        """
        if np.ndim(theta) != 0:
            raise NotImplementedError()
        return sps.gamma.logpdf(theta, a=self.a, scale=self.scale, loc=self.loc)

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        return self.rng.gamma(shape=self.a, scale=self.scale, size=n_samples)

    def _gradient(self, theta: float) -> float:
        """As computed by Wolfram Alpha.

        Parameters
        ----------
        theta: float
            A hyperparameter configuration

        Returns
        -------
        float
        """
        if np.ndim(theta) == 0:
            # Multiply by theta because of the chain rule...
            return ((self.a - 1) / theta - (1 / self.scale)) * theta
        else:
            raise NotImplementedError()
