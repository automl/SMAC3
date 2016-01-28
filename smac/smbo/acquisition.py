# encoding=utf8
import logging
from scipy.stats import norm
import numpy as np

__author__ = "Aaron Klein"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"

class BestObservation(object):
    """
    Container class for storing information about the best observation.
    """

    def __init__(self, model, X_lower, X_upper):
        self.model = model
        self.X_upper = X_upper
        self.X_lower = X_lower

    def estimate_incumbent(self):
        best = np.argmin(self.model.Y)
        incumbent = self.model.X[best]
        incumbent_value = self.model.Y[best]

        return incumbent[np.newaxis, :], incumbent_value[:, np.newaxis]


class AcquisitionFunction(object):
    long_name = ""

    def __str__(self):
        return type(self).__name__ + " (" + self.long_name + ")"

    def __init__(self, model, X_lower, X_upper, **kwargs):
        """
        A base class for acquisition functions.

        Parameters
        ----------
        model : Model object
            Models the objective function.
        X_lower : (D) numpy array
            Specified the lower bound of the input space. Each entry
            corresponds to one dimension.
        X_upper : (D) numpy array
            Specified the upper bound of the input space. Each entry
            corresponds to one dimension.
        """
        self.model = model
        self.X_lower = X_lower
        self.X_upper = X_upper

        self.logger = logging.getLogger("AcquisitionFunction")

        assert np.any(self.X_lower < self.X_upper)

    def update(self, model):
        """
        This method will be called if the model is updated. E.g.
        Entropy search uses it to update it's approximation of P(x=x_min)

        Parameters
        ----------
        model : Model object
            Models the objective function.
        """

        self.model = model

    def __call__(self, X, derivative=False):
        """
        Computes the acquisition value for a given point X

        Parameters
        ----------
        X: np.ndarray(1, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition
            function at X is returned
        """
        if np.any(X < self.X_lower) or np.any(X > self.X_upper):
            ValueError("Test point is out of bounds")

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        if derivative:
            acq, grad = zip(
                *[self.compute(x[np.newaxis, :], derivative) for x in X])
            acq = np.array(acq)[:, :, 0]
            grad = np.array(grad)[:, :, 0]

            if np.any(np.isnan(acq)):
                idx = np.where(np.isnan(acq))[0]
                acq[idx, :] = -np.finfo(np.float).max
                grad[idx, :] = -np.inf
            return acq, grad

        else:
            acq = [self.compute(x[np.newaxis, :], derivative) for x in X]
            acq = np.array(acq)[:, :, 0]
            if np.any(np.isnan(acq)):
                idx = np.where(np.isnan(acq))[0]
                acq[idx, :] = -np.finfo(np.float).max
            return acq

    def compute(self, X, derivative=False):
        """
        Computes the acquisition value for a given point X. This function has
        to be overwritten in a derived class.

        Parameters
        ----------
        X: np.ndarray(1, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition
            function at X is returned
        """
        raise NotImplementedError()


class EI(AcquisitionFunction):

    def __init__(self,
                 model,
                 X_lower,
                 X_upper,
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

        super(EI, self).__init__(model, X_lower, X_upper)
        self.par = par
        self.rec = BestObservation(self.model, self.X_lower, self.X_upper)

    def update(self, model):
        """
        This method will be called if the model is updated.
        Parameters
        ----------
        model : Model object
            Models the objective function.
        """

        super(EI, self).update(model)
        self.rec = BestObservation(self.model, self.X_lower, self.X_upper)

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
        _, eta = self.rec.estimate_incumbent()

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
                self.logger.error("Expected Improvement is smaller than 0!")
                raise ValueError

        if derivative:
            return f, df
        else:
            return f
