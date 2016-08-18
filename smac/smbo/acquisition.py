# encoding=utf8
import abc
import logging
from scipy.stats import norm
import numpy as np

__author__ = "Aaron Klein"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


class AbstractAcquisitionFunction(object):
    __metaclass__ = abc.ABCMeta
    long_name = ""

    def __str__(self):
        return type(self).__name__ + " (" + self.long_name + ")"

    def __init__(self, model, **kwargs):
        """
        A base class for acquisition functions.

        Parameters
        ----------
        model : Model object
            Models the objective function.

        """
        self.model = model

        self.logger = logging.getLogger("AcquisitionFunction")

    def update(self, **kwargs):
        """Update the acquisition functions values.

        This method will be called if the model is updated. E.g.
        Entropy search uses it to update it's approximation of P(x=x_min),
        EI uses it to update the current fmin.

        The default implementation takes all keyword arguments and sets the
        respective attributes for the acquisition function object.

        Parameters
        ----------
        kwargs
        """

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __call__(self, X, derivative=False):
        """
        Computes the acquisition value for a given point X

        Parameters
        ----------
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative : Boolean
            If is set to true also the derivative of the acquisition
            function at X is returned
        """

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        acq = self._compute(X, derivative)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(np.float).max
        return acq

    @abc.abstractmethod
    def _compute(self, X, derivative=False):
        """
        Computes the acquisition value for a given point X. This function has
        to be overwritten in a derived class.

        Parameters
        ----------
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative : Boolean
            If is set to true also the derivative of the acquisition
            function at X is returned

        Returns
        -------
        np.ndarray :
        """
        raise NotImplementedError()


class EI(AbstractAcquisitionFunction):

    def __init__(self,
                 model,
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
        par: float
            Controls the balance between exploration
            and exploitation of the acquisition function. Default is 0.01
        """

        super(EI, self).__init__(model)
        self.long_name = 'Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X, derivative=False, **kwargs):
        """
        Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            If is set to true also the derivative of the acquisition
            function at X is returned

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        np.ndarray(N,1)
            Derivative of Expected Improvement at X (only if derivative=True)
        """

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        assert m.shape[1] == 1
        assert v.shape[1] == 1
        s = np.sqrt(v)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        z = (self.eta - m - self.par) / s
        f = (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)
        f[s == 0.0] = 0.0

        if derivative:
            dmdx, ds2dx = self.model.predictive_gradients(X)
            dmdx = dmdx[0]
            ds2dx = ds2dx[0][:, None]
            dsdx = ds2dx / (2 * s)
            df = (-dmdx * norm.cdf(z) + (dsdx * norm.pdf(z))).T
            df[s == 0.0] = 0.0

        if (f < 0).any():
            self.logger.error("Expected Improvement is smaller than 0!")
            raise ValueError

        if derivative:
            return f, df
        else:
            return f


class EIPS(EI):
    def __init__(self,
                 model,
                 par=0.01,
                 **kwargs):
        r"""
        Computes for a given x the expected improvement as
        acquisition value.
        :math:`EI(X) :=
            \frac{\mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) -
                  f_{t+1}(\mathbf{X}) - \xi\right] \} ]}
                  {np.log10(r(x))}`,
        with :math:`f(X^+)` as the incumbent and :math:`r(x)` as runtime.

        Parameters
        ----------
        model: Model object
            A model that implements at least
                 - predict(X)
                 - getCurrentBestX().
            If you want to calculate derivatives than it should also support
                 - predictive_gradients(X)
        par: float
            Controls the balance between exploration
            and exploitation of the acquisition function. Default is 0.01
        """

        super(EIPS, self).__init__(model)
        self.long_name = 'Expected Improvement per Second'

    def _compute(self, X, derivative=False, **kwargs):
        """
        Computes the EIPS value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        derivative: Boolean
            Raises NotImplementedError if True.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement per Second of X
        """

        if derivative:
            raise NotImplementedError()

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        assert m.shape[1] == 2
        assert v.shape[1] == 2
        m_cost = m[:, 0]
        v_cost = v[:, 0]
        # The model already predicts log(runtime)
        m_runtime = m[:, 1]
        s = np.sqrt(v_cost)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        z = (self.eta - m_cost - self.par) / s
        f = (self.eta - m_cost - self.par) * norm.cdf(z) + s * norm.pdf(z)
        f = f / m_runtime
        f[s == 0.0] = 0.0

        if (f < 0).any():
            self.logger.error("Expected Improvement per Second is smaller than "
                              "0!")
            raise ValueError

        return f.reshape((-1, 1))