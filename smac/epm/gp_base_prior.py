
import numpy as np
import scipy.stats as sps


class BasePrior(object):

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


class TophatPrior(BasePrior):

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
            raise Exception("Upper bound of Tophat prior must be greater \
            than the lower bound!")

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

        if np.any(theta < self.min) or np.any(theta > self.max):
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
        return np.zeros([theta.shape[0]])


class HorseshoePrior(BasePrior):

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
        # We computed it exactly as in the original spearmint code
        if np.any(theta == 0.0):
            return np.inf
        return np.log(np.log(1 + 3.0 * (self.scale / np.exp(theta)) ** 2))

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
        a = -(6 * self.scale ** 2)
        b = (3 * self.scale ** 2 + np.exp(2 * theta))
        b *= np.log(3 * self.scale ** 2 * np.exp(- 2 * theta) + 1)
        return a / b


class LognormalPrior(BasePrior):
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

        self.sigma = sigma
        self.mean = mean

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

        return sps.lognorm.logpdf(theta, self.sigma, loc=self.mean)

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
        pass


class NormalPrior(BasePrior):
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
