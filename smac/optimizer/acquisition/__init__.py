# encoding=utf8
import abc
from typing import Any, List, Tuple

import copy

import numpy as np
from ConfigSpace.hyperparameters import FloatHyperparameter
from scipy.stats import norm

from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM
from smac.utils.logging import PickableLoggerAdapter

__author__ = "Aaron Klein, Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class AbstractAcquisitionFunction(object, metaclass=abc.ABCMeta):
    """Abstract base class for acquisition function.

    Parameters
    ----------
    model : AbstractEPM
        Models the objective function.

    Attributes
    ----------
    model
    logger
    """

    def __init__(self, model: AbstractEPM):
        self.model = model
        self._required_updates = ("model",)  # type: Tuple[str, ...]
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
                    "Acquisition function %s needs to be updated with key %s, but only got "
                    "keys %s." % (self.__class__.__name__, key, list(kwargs.keys()))
                )
        for key in kwargs:
            if key in self._required_updates:
                setattr(self, key, kwargs[key])

    def __call__(self, configurations: List[Configuration]) -> np.ndarray:
        """Computes the acquisition value for a given X.

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
            acq[idx, :] = -np.finfo(float).max
        return acq

    @abc.abstractmethod
    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the acquisition value for a given point X. This function has to be overwritten
        in a derived class.

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
        """Constructor.

        Parameters
        ----------
        model : AbstractEPM
            The model needs to implement an additional attribute ``models`` which contains the different models to
            integrate over.
        kwargs
            Additional keyword arguments
        """
        super().__init__(model)
        self.long_name = "Integrated Acquisition Function (%s)" % acquisition_function.__class__.__name__
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
        model = kwargs["model"]
        del kwargs["model"]
        if not hasattr(model, "models") or len(model.models) == 0:
            raise ValueError("IntegratedAcquisitionFunction requires at least one model to integrate!")
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
            raise ValueError("Need to call update first!")
        return np.array([func._compute(X) for func in self._functions]).mean(axis=0)


class PriorAcquisitionFunction(AbstractAcquisitionFunction):
    r"""Weight the acquisition function with a user-defined prior over the optimum.

    See "PiBO: Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization" by Carl
    Hvarfner et al. (###nolinkyet###) for further details.
    """

    def __init__(
        self,
        model: AbstractEPM,
        acquisition_function: AbstractAcquisitionFunction,
        decay_beta: float,
        prior_floor: float = 1e-12,
        discretize: bool = False,
        discrete_bins_factor: float = 10.0,
        **kwargs: Any,
    ):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            Models the objective function.
        decay_beta: Decay factor on the user prior - defaults to n_iterations / 10 if not specifed
            otherwise.
        prior_floor : Lowest possible value of the prior, to ensure non-negativity for all values
            in the search space.
        discretize : Whether to discretize (bin) the densities for continous parameters. Triggered
            for Random Forest models and continous hyperparameters to avoid a pathological case
            where all Random Forest randomness is removed (RF surrogates require piecewise constant
            acquisition functions to be well-behaved)
        discrete_bins_factor : If discretizing, the multiple on the number of allowed bins for
            each parameter

        kwargs
            Additional keyword arguments
        """
        super().__init__(model)
        self.long_name = "Prior Acquisition Function (%s)" % acquisition_function.__class__.__name__
        self.acq = acquisition_function
        self._functions = []  # type: List[AbstractAcquisitionFunction]
        self.eta = None
        self.hyperparameters = self.model.get_configspace().get_hyperparameters_dict()
        self.decay_beta = decay_beta
        self.prior_floor = prior_floor
        self.discretize = discretize
        if self.discretize:
            self.discrete_bins_factor = discrete_bins_factor

        # check if the acquisition function is LCB or TS - then the acquisition function values
        # need to be rescaled to assure positiveness & correct magnitude
        if isinstance(self.acq, IntegratedAcquisitionFunction):
            acquisition_type = self.acq.acq
        else:
            acquisition_type = self.acq

        self.rescale_acq = isinstance(acquisition_type, (LCB, TS))
        self.iteration_number = 0

    def update(self, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        Updates the model, the accompanying acquisition function and tracks the iteration number.

        Parameters
        ----------
        kwargs
            Additional keyword arguments
        """
        self.iteration_number += 1
        self.acq.update(**kwargs)
        self.eta = kwargs.get("eta")

    def _compute_prior(self, X: np.ndarray) -> np.ndarray:
        """Computes the prior-weighted acquisition function values, where the prior on each
        parameter is multiplied by a decay factor controlled by the parameter decay_beta and
        the iteration number. Multivariate priors are not supported, for now.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the user-specified prior
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            The user prior over the optimum for values of X
        """
        prior_values = np.ones((len(X), 1))
        # iterate over the hyperparmeters (alphabetically sorted) and the columns, which come
        # in the same order
        for parameter, X_col in zip(self.hyperparameters.values(), X.T):
            if self.discretize and isinstance(parameter, FloatHyperparameter):
                number_of_bins = int(np.ceil(self.discrete_bins_factor * self.decay_beta / self.iteration_number))
                prior_values *= self._compute_discretized_pdf(parameter, X_col, number_of_bins) + self.prior_floor
            else:
                prior_values *= parameter._pdf(X_col[:, np.newaxis])

        return prior_values

    def _compute_discretized_pdf(
        self, parameter: FloatHyperparameter, X_col: np.ndarray, number_of_bins: int
    ) -> np.ndarray:
        """Discretizes (bins) prior values on continous a specific continous parameter
        to an increasingly coarse discretization determined by the prior decay parameter.

        Parameters
        ----------
        parameter: a FloatHyperparameter that, due to using a random forest
            surrogate, must have its prior discretized
        X_col: np.ndarray(N, ), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, ), with N as
            the number of points to evaluate for the specific hyperparameter
        number_of_bins: The number of unique values allowed on the
            discretized version of the pdf.

        Returns
        -------
        np.ndarray(N,1)
            The user prior over the optimum for the parameter at hand.
        """
        # evaluates the actual pdf on all the relevant points
        pdf_values = parameter._pdf(X_col[:, np.newaxis])
        # retrieves the largest value of the pdf in the domain
        lower, upper = (0, parameter.get_max_density())
        # creates the bins (the possible discrete options of the pdf)
        bin_values = np.linspace(lower, upper, number_of_bins)
        # generates an index (bin) for each evaluated point
        bin_indices = np.clip(
            np.round((pdf_values - lower) * number_of_bins / (upper - lower)), 0, number_of_bins - 1
        ).astype(int)
        # gets the actual value for each point
        prior_values = bin_values[bin_indices]
        return prior_values

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the prior-weighted acquisition function values, where the prior on each
        parameter is multiplied by a decay factor controlled by the parameter decay_beta and
        the iteration number. Multivariate priors are not supported, for now.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Prior-weighted acquisition function values of X
        """
        if self.rescale_acq:
            # for TS and UCB, we need to scale the function values to not run into issues
            # of negative values or issues of varying magnitudes (here, they are both)
            # negative by design and just flipping the sign leads to picking the worst point)
            acq_values = np.clip(self.acq._compute(X) + self.eta, 0, np.inf)
        else:
            acq_values = self.acq._compute(X)
        prior_values = self._compute_prior(X) + self.prior_floor
        decayed_prior_values = np.power(prior_values, self.decay_beta / self.iteration_number)

        return acq_values * decayed_prior_values


class EI(AbstractAcquisitionFunction):
    r"""Computes for a given x the expected improvement as
    acquisition value.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the best location.
    """

    def __init__(self, model: AbstractEPM, par: float = 0.0):
        """Constructor.

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
        self.long_name = "Expected Improvement"
        self.par = par
        self.eta = None
        self._required_updates = ("model", "eta")

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
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

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
            raise ValueError("Expected Improvement is smaller than 0 for at least one " "sample.")

        return f


class EIPS(EI):
    def __init__(self, model: AbstractEPM, par: float = 0.0):
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
        self.long_name = "Expected Improvement per Second"

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
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

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
            raise ValueError("Expected Improvement per Second is smaller than 0 " "for at least one sample.")

        return f.reshape((-1, 1))


class LogEI(AbstractAcquisitionFunction):
    def __init__(self, model: AbstractEPM, par: float = 0.0):
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
        self.long_name = "Expected Improvement"
        self.par = par
        self.eta = None
        self._required_updates = ("model", "eta")

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
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)

        def calculate_log_ei():
            # we expect that f_min is in log-space
            f_min = self.eta - self.par
            v = (f_min - m) / std
            return (np.exp(f_min) * norm.cdf(v)) - (np.exp(0.5 * var_ + m) * norm.cdf(v - std))

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
            raise ValueError("Expected Improvement is smaller than 0 for at least one sample.")

        return log_ei.reshape((-1, 1))


class PI(AbstractAcquisitionFunction):
    def __init__(self, model: AbstractEPM, par: float = 0.0):
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
        self.long_name = "Probability of Improvement"
        self.par = par
        self.eta = None
        self._required_updates = ("model", "eta")

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
            raise ValueError(
                "No current best specified. Call update("
                "eta=<float>) to inform the acquisition function "
                "about the current best value."
            )

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        return norm.cdf((self.eta - m - self.par) / std)


class LCB(AbstractAcquisitionFunction):
    def __init__(self, model: AbstractEPM, par: float = 1.0):
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
        self.long_name = "Lower Confidence Bound"
        self.par = par
        self.num_data = None
        self._required_updates = ("model", "num_data")

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
            raise ValueError(
                "No current number of Datapoints specified. Call update("
                "num_data=<int>) to inform the acquisition function "
                "about the number of datapoints."
            )
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        beta = 2 * np.log((X.shape[1] * self.num_data**2) / self.par)
        return -(m - np.sqrt(beta) * std)


class TS(AbstractAcquisitionFunction):
    def __init__(self, model: AbstractEPM, par: float = 0.0):
        r"""Do a Thompson Sampling for a given x over the best so far value as
        acquisition value.

        Warning
        -------
        Thompson Sampling can only be used together with
        smac.optimizer.ei_optimization.RandomSearch, please do not use
        smac.optimizer.ei_optimization.LocalAndSortedRandomSearch to optimize TS
        acquisition function!

        :math:`TS(X) ~ \mathcal{N}(\mu(\mathbf{X}),\sigma(\mathbf{X}))'
        Returns -TS(X) as the acquisition_function optimizer maximizes the acquisition value.

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            TS does not require par here, we only wants to make it consistent with
            other acquisition functions.
        """
        super(TS, self).__init__(model)
        self.long_name = "Thompson Sampling"
        self.par = par
        self.num_data = None
        self._required_updates = ("model",)

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Sample a new value from a gaussian distribution whose mean and covariance values
        are given by model.

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
            return -sample_function(X, n_funcs=1)

        m, var_ = self.model.predict_marginalized_over_instances(X)
        rng = getattr(self.model, "rng", np.random.RandomState(self.model.seed))
        m = m.flatten()
        var_ = np.diag(var_.flatten())
        return -rng.multivariate_normal(m, var_, 1).T
