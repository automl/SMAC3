# encoding=utf8
import abc
import copy
from typing import List, Any, Tuple

import numpy as np
from scipy.stats import norm

from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM
from smac.utils.logging import PickableLoggerAdapter

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

    def __init__(self, model: AbstractEPM):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            Models the objective function.
        """
        self.model = model
        self._required_updates = ('model', )  # type: Tuple[str, ...]
        self.logger = PickableLoggerAdapter(self.__module__ + "." + self.__class__.__name__)

    def update(self, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        This method will be called after fitting the model, but before maximizing the acquisition
        function. As an examples, EI uses it to update the current fmin.

        The default implementation only updates the attributes of the acqusition function which
        are already present.

        Parameters
        ----------
        kwargs
        """
        for key in self._required_updates:
            if key not in kwargs:
                raise ValueError(
                    'Acquisition function %s needs to be updated with key %s, but only got '
                    'keys %s.'
                    % (self.__class__.__name__, key, list(kwargs.keys()))
                )
        for key in kwargs:
            if key in self._required_updates:
                setattr(self, key, kwargs[key])

    def __call__(self, configurations: List[Configuration]) -> np.ndarray:
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
    def _compute(self, X: np.ndarray) -> np.ndarray:
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


class IntegratedAcquisitionFunction(AbstractAcquisitionFunction):

    r"""Marginalize over Model hyperparameters to compute the integrated acquisition function.

    See "Practical Bayesian Optimization of Machine Learning Algorithms" by Jasper Snoek et al.
    (https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
    for further details.
    """

    def __init__(self, model: AbstractEPM, acquisition_function: AbstractAcquisitionFunction, **kwargs: Any):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            The model needs to implement an additional attribute ``models`` which contains the different models to
            integrate over.
        kwargs
            Additional keyword arguments
        """

        super().__init__(model)
        self.long_name = 'Integrated Acquisition Function (%s)' % acquisition_function.__class__.__name__
        self.acq = acquisition_function
        self._functions = []  # type: List[AbstractAcquisitionFunction]
        self.eta = None

    def update(self, **kwargs: Any) -> None:
        """Update the acquisition functions values.

        This method will be called if the model is updated. E.g. entropy search uses it to update its approximation
        of P(x=x_min), EI uses it to update the current fmin.

        This implementation creates an acquisition function object for each model to integrate over and sets the
        respective attributes for each acquisition function object.

        Parameters
        ----------
        model : AbstractEPM
            The model needs to implement an additional attribute ``models`` which contains the different models to
            integrate over.
        kwargs
        """
        model = kwargs['model']
        del kwargs['model']
        if not hasattr(model, 'models') or len(model.models) == 0:
            raise ValueError('IntegratedAcquisitionFunction requires at least one model to integrate!')
        if len(self._functions) == 0 or len(self._functions) != len(model.models):
            self._functions = [copy.deepcopy(self.acq) for _ in model.models]
        for submodel, func in zip(model.models, self._functions):
            func.update(model=submodel, **kwargs)

    def _compute(self, X: np.ndarray) -> np.ndarray:
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
        if self._functions is None:
            raise ValueError('Need to call update first!')
        return np.array([func._compute(X) for func in self._functions]).mean(axis=0)


class EI(AbstractAcquisitionFunction):

    r"""Computes for a given x the expected improvement as
    acquisition value.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the best location.
    """

    def __init__(self,
                 model: AbstractEPM,
                 par: float = 0.0):
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
        self._required_updates = ('model', 'eta')

    def _compute(self, X: np.ndarray) -> np.ndarray:
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
                 par: float = 0.0):
        r"""Computes for a given x the expected improvement as
        acquisition value.
        :math:`EI(X) := \frac{\mathbb{E}\left[\max\{0,f(\mathbf{X^+})-f_{t+1}(\mathbf{X})-\xi\right]\}]}{np.log(r(x))}`,
        with :math:`f(X^+)` as the best location and :math:`r(x)` as runtime.

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

    def _compute(self, X: np.ndarray) -> np.ndarray:
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
                 par: float = 0.0):
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
        self._required_updates = ('model', 'eta')

    def _compute(self, X: np.ndarray) -> np.ndarray:
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


class PI(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractEPM,
                 par: float = 0.0):
        r"""Computes the probability of improvement for a given x over the best so far value as acquisition value.

        :math:`P(f_{t+1}(\mathbf{X})\geq f(\mathbf{X^+}))` :math:`:= \Phi(\\frac{ \mu(\mathbf{X})-f(\mathbf{X^+}) }
        { \sigma(\mathbf{X}) })` with :math:`f(X^+)` as the incumbent and :math:`\Phi` the cdf of the standard normal

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(PI, self).__init__(model)
        self.long_name = 'Probability of Improvement'
        self.par = par
        self.eta = None
        self._required_updates = ('model', 'eta')

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the PI value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate PI. N is the number of points and D the dimension for the points

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<float>) to inform the acquisition function '
                             'about the current best value.')

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        return norm.cdf((self.eta - m - self.par) / std)


class LCB(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractEPM,
                 par: float = 1.0):
        r"""Computes the lower confidence bound for a given x over the best so far value as
        acquisition value.

        :math:`LCB(X) = \mu(\mathbf{X}) - \sqrt(\beta_t)\sigma(\mathbf{X})`

        Returns -LCB(X) as the acquisition_function optimizer maximizes the acquisition value.

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=1.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(LCB, self).__init__(model)
        self.long_name = 'Lower Confidence Bound'
        self.par = par
        self.num_data = None
        self._required_updates = ('model', 'num_data')

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the LCB value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate LCB. N is the number of points and D the dimension for the points

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if self.num_data is None:
            raise ValueError('No current number of Datapoints specified. Call update('
                             'num_data=<int>) to inform the acquisition function '
                             'about the number of datapoints.')
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        beta = 2 * np.log((X.shape[1] * self.num_data**2) / self.par)
        return -(m - np.sqrt(beta) * std)


class TS(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractEPM,
                 par: float = 0.0):
        r"""Do a Thompson Sampling for a given x over the best so far value as
        acquisition value.

        Thompson Sampling can only be used together with smac.optimizer.ei_optimization.RandomSearch, please do not
        use smac.optimizer.ei_optimization.LocalAndSortedRandomSearch to optimize TS acquisition function!!!

        :math:`TS(X) ~ \mathcal{N}(\mu(\mathbf{X}),\sigma(\mathbf{X}))'
        Returns -TS(X) as the acquisition_function optimizer maximizes the acquisition value.
        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            TS does not require par here, we only wants to make it consistent with other acquisition functions
        """
        super(TS, self).__init__(model)
        self.long_name = 'Thompson Sampling'
        self.par = par
        self.num_data = None
        self._required_updates = ('model', )

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Sample a new value from a gaussian distribution whose mean and covariance values are given by model
        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to be evaluated where we could sample a value. N is the number of points and D the dimension
           for the points
        Returns
        -------
        np.ndarray(N,1)
            negative sample value of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        sample_function = getattr(self.model, "sample_functions", None)
        if callable(sample_function):
            return - sample_function(X, n_funcs=1)

        m, var_ = self.model.predict_marginalized_over_instances(X)
        rng = getattr(self.model, 'rng', np.random.RandomState(self.model.seed))
        m = m.flatten()
        var_ = np.diag(var_.flatten())
        return - rng.multivariate_normal(m, var_, 1).T
