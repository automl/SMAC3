# encoding=utf8
import logging
from scipy.stats import norm
import numpy as np

from robo.acquisition.base import AcquisitionFunction

logger = logging.getLogger(__name__)


class EI(AcquisitionFunction):

    def __init__(
            self,
            model,
            X_lower,
            X_upper,
            compute_incumbent,
            par=0.01,
            **kwargs):

        r"""
        Computes for a given x the expected improvement as
        acquisition value.
        :math:`EI(X) :=
            \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) -
                f_{t+1}(\mathbf{X}) - \xi\right] \} ]`, with
        :math:`f(X^+)` as the incumbent.

        Parameters
        ----------
        model: Model object
            A model that implements at least
                 - predict(X)
                 - getCurrentBestX().
            If you want to calculate derivatives than it should also support
                 - predictive_gradients(X)

        X_lower: np.ndarray (D)
            Lower bounds of the input space
        X_upper: np.ndarray (D)
            Upper bounds of the input space
        compute_incumbent: func
            A python function that takes as input a model and returns
            a np.array as incumbent
        par: float
            Controls the balance between exploration
            and exploitation of the acquisition function. Default is 0.01
        """

        self.par = par
        self.compute_incumbent = compute_incumbent
        super(EI, self).__init__(model, X_lower, X_upper)

        logger.debug("Test")

    def compute(self, X, derivative=False, **kwargs):
        """
        Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(1, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition
            function at X is returned

        Returns
        -------
        np.ndarray(1,1)
            Expected Improvement of X
        np.ndarray(1,D)
            Derivative of Expected Improvement at X (only if derivative=True)
        """

        if X.shape[0] > 1:
            raise ValueError("EI is only for single test points")

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            if derivative:
                f = 0
                df = np.zeros((1, X.shape[1]))
                return np.array([[f]]), np.array([df])
            else:
                return np.array([[0]])

        m, v = self.model.predict(X, full_cov=True)

        # Use the best seen observation as incumbent
        _, eta = self.compute_incumbent(self.model)

        s = np.sqrt(v)

        if (s == 0).any():
            f = np.array([[0]])
            df = np.zeros((1, X.shape[1]))

        else:
            z = (eta - m - self.par) / s
            f = (eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)
            if derivative:
                dmdx, ds2dx = self.model.predictive_gradients(X)
                dmdx = dmdx[0]
                ds2dx = ds2dx[0][:, None]
                dsdx = ds2dx / (2 * s)
                df = (-dmdx * norm.cdf(z) + (dsdx * norm.pdf(z))).T
            if (f < 0).any():
                logger.error("Expected Improvement is smaller than 0!")
                raise Exception
            if len(f.shape) == 1:
                f = np.array([f])

        if derivative:
            if len(df.shape) == 3:
                return_df = df
            else:
                return_df = np.array([df])
            return f, return_df
        else:
            return f
