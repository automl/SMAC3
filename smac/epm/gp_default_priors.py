
import numpy as np

from smac.epm.gp_base_prior import Prior, TophatPrior, \
    LognormalPrior, HorseshoePrior, SoftTopHatPrior


class DefaultPrior(Prior):

    def __init__(self, n_dims: int, rng: np.random.RandomState=None):
        """
        This class is a verbatim copy of the implementation of RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop
        """
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        # The number of hyperparameters
        self.n_dims = n_dims

        # Prior for the Matern52 lengthscales
        self.tophat = TophatPrior(-10 - 1e-10, 2 + 1e-10, rng=self.rng)

        # Prior for the covariance amplitude
        self.ln_prior = LognormalPrior(mean=0.0, sigma=1.0, rng=self.rng)

        # Prior for the noise
        self.horseshoe = HorseshoePrior(scale=0.1, rng=self.rng)
        self.noise_bound = SoftTopHatPrior(lower_bound=np.log(1e-10), rng=self.rng)

    def lnprob(self, theta: np.ndarray):
        lp = 0
        # Covariance amplitude
        lp += self.ln_prior.lnprob(theta[0])
        # Lengthscales
        lp += self.tophat.lnprob(theta[1:-1])
        # Noise
        lp += self.horseshoe.lnprob(theta[-1])
        lp += self.noise_bound.lnprob(theta[-1])

        return lp

    def sample_from_prior(self, n_samples: int):
        p0 = np.zeros([n_samples, self.n_dims])
        # Covariance amplitude
        p0[:, 0] = self.ln_prior.sample_from_prior(n_samples)[:, 0]
        # Lengthscales
        ls_sample = np.array([self.tophat.sample_from_prior(n_samples)[:, 0]
                              for _ in range(1, (self.n_dims - 1))]).T
        p0[:, 1:(self.n_dims - 1)] = ls_sample
        # Noise
        p0[:, -1] = self.horseshoe.sample_from_prior(n_samples)[:, 0]
        return list(p0)

    def gradient(self, theta: np.ndarray):
        grad = np.zeros([theta.shape[0]])
        # Covariance amplitude
        grad[0] += self.ln_prior.gradient(theta[0])
        # Lengthscales
        grad[1: -1] += self.tophat.gradient(theta[1: -1])
        # Noise
        grad[-1] += self.horseshoe.gradient(theta[-1])
        grad[-1] += self.noise_bound.gradient(theta[-1])
        return grad
