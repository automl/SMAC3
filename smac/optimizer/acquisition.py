# encoding=utf8
import abc
import logging
from typing import List

import numpy as np
from scipy.stats import norm

from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM

__author__ = "Aaron Klein, Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class AbstractAcquisitionFunction(object, metaclass=abc.ABCMeta):
    """Abstract base class for acquisition function

    Attributes
    ----------
    model
    logger
    """

    def __str__(self):
        return type(self).__name__ + " (" + self.long_name + ")"

    def __init__(self, model: AbstractEPM, **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            Models the objective function.
        """
        self.model = model
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

    def update(self, **kwargs):
        """Update the acquisition functions values.

        This method will be called if the model is updated. E.g.
        entropy search uses it to update its approximation of P(x=x_min),
        EI uses it to update the current fmin.

        The default implementation takes all keyword arguments and sets the
        respective attributes for the acquisition function object.

        Parameters
        ----------
        kwargs
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __call__(self, configurations: List[Configuration]):
        """Computes the acquisition value for a given X

        Parameters
        ----------
        configurations : list
            The configurations where the acquisition function
            should be evaluated. 

        Returns
        -------
        np.ndarray(N, 1)
            acquisition values for X
        """
        X = convert_configurations_to_array(configurations)
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        acq = self._compute(X)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(np.float).max
        return acq

    @abc.abstractmethod
    def _compute(self, X: np.ndarray):
        """Computes the acquisition value for a given point X. This function has
        to be overwritten in a derived class.

        Parameters
        ----------
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Acquisition function values wrt X
        """
        raise NotImplementedError()


class EI(AbstractAcquisitionFunction):

    r"""Computes for a given x the expected improvement as
    acquisition value.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the incumbent.
    """

    def __init__(self,
                 model: AbstractEPM,
                 par: float=0.0,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """

        super(EI, self).__init__(model)
        self.long_name = 'Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.eta - m - self.par) / s
            return (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()
        if (f < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one "
                "sample.")

        return f


class EIPS(EI):
    def __init__(self,
                 model: AbstractEPM,
                 par: float=0.0,
                 **kwargs):
        r"""Computes for a given x the expected improvement as
        acquisition value.
        :math:`EI(X) := \frac{\mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi\right] \} ]} {np.log(r(x))}`,
        with :math:`f(X^+)` as the incumbent and :math:`r(x)` as runtime.

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X) returning a tuples of
                   predicted cost and running time
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(EIPS, self).__init__(model, par=par)
        self.long_name = 'Expected Improvement per Second'

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EIPS value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement per Second of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        if m.shape[1] != 2:
            raise ValueError("m has wrong shape: %s != (-1, 2)" % str(m.shape))
        if v.shape[1] != 2:
            raise ValueError("v has wrong shape: %s != (-1, 2)" % str(v.shape))

        m_cost = m[:, 0]
        v_cost = v[:, 0]
        # The model already predicts log(runtime)
        m_runtime = m[:, 1]
        s = np.sqrt(v_cost)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.eta - m_cost - self.par) / s
            f = (self.eta - m_cost - self.par) * norm.cdf(z) + s * norm.pdf(z)
            f = f / m_runtime
            return f

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()

        if (f < 0).any():
            raise ValueError(
                "Expected Improvement per Second is smaller than 0 "
                "for at least one sample.")

        return f.reshape((-1, 1))


class LogEI(AbstractAcquisitionFunction):

    def __init__(self,
                 model: AbstractEPM,
                 par: float=0.0,
                 **kwargs):
        r"""Computes for a given x the logarithm expected improvement as
        acquisition value.

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(LogEI, self).__init__(model)
        self.long_name = 'Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)

        def calculate_log_ei():
            # we expect that f_min is in log-space
            f_min = self.eta - self.par
            v = (f_min - m) / std
            return (np.exp(f_min) * norm.cdf(v)) - \
                (np.exp(0.5 * var_ + m) * norm.cdf(v - std))

        if np.any(std == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            std_copy = np.copy(std)
            std[std_copy == 0.0] = 1.0
            log_ei = calculate_log_ei()
            log_ei[std_copy == 0.0] = 0.0
        else:
            log_ei = calculate_log_ei()

        if (log_ei < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one sample.")

        return log_ei.reshape((-1, 1))
