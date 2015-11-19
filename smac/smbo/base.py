# encoding=utf8

import numpy as np


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

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        if derivative:
            acq, grad = self.compute(X, derivative)
            if np.isnan(acq):
                return np.array([[-np.finfo(np.float).max]]
                                ), np.array([[-np.inf]])
            else:
                return acq, grad
        else:
            acq = self.compute(X, derivative)
            if np.isnan(acq):
                return np.array([[-np.finfo(np.float).max]])
            else:
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
