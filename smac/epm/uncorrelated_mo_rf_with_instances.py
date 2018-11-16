import numpy as np

from smac.epm.base_epm import AbstractEPM
from smac.epm.rf_with_instances import RandomForestWithInstances

from typing import List, Dict, Any, Optional


class UncorrelatedMultiObjectiveRandomForestWithInstances(AbstractEPM):

    """Wrapper for the random forest to predict multiple targets.

    Only the a list with the target names and the types array for the
    underlying forest model are mandatory. All other hyperparameters to
    the random forest can be passed via kwargs. Consult the documentation of
    the random forest for the hyperparameters and their meanings.

    Attributes
    ----------
    target_names
    num_targets
    estimators
    """

    def __init__(
            self,
            target_names: List[str],
            bounds: List[float],
            types: np.ndarray,
            rf_kwargs: Optional[Dict[str, Any]]=None,
            **kwargs
    ):
        """Constructor

        Parameters
        ----------
        target_names : list
            List of str, each entry is the name of one target dimension. Length
            of the list will be ``n_objectives``.

        bounds : np.ndarray
            See :class:`~smac.epm.rf_with_instances.RandomForestWithInstances` documentation.

        types : np.ndarray
            See :class:`~smac.epm.rf_with_instances.RandomForestWithInstances` documentation.

        kwargs
            See :class:`~smac.epm.rf_with_instances.RandomForestWithInstances` documentation.
        """
        super().__init__(bounds=bounds, types=types, **kwargs)
        if rf_kwargs is None:
            rf_kwargs = {}

        self.target_names = target_names
        self.num_targets = len(self.target_names)
        self.estimators = [RandomForestWithInstances(types, bounds, **rf_kwargs)
                           for i in range(self.num_targets)]

    def _train(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, n_objectives]
            The corresponding target values. n_objectives must match the
            number of target names specified in the constructor.

        Returns
        -------
        self
        """
        for i, estimator in enumerate(self.estimators):
            estimator.train(X, Y[:, i], **kwargs)

        return self

    def _predict(self, X: np.ndarray):
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance
        features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, n_objectives]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, n_objectives]
            Predictive variance
        """
        mean = np.zeros((X.shape[0], self.num_targets))
        var = np.zeros((X.shape[0], self.num_targets))
        for i, estimator in enumerate(self.estimators):
            m, v = estimator.predict(X)
            mean[:, i] = m.flatten()
            var[:, i] = v.flatten()
        return mean, var

    def predict_marginalized_over_instances(self, X: np.ndarray):
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Parameters
        ----------
        X : np.ndarray of shape = [n_features (config), ]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, n_objectives]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, n_objectives]
            Predictive variance
        """
        mean = np.zeros((X.shape[0], self.num_targets))
        var = np.zeros((X.shape[0], self.num_targets))
        for i, estimator in enumerate(self.estimators):
            m, v = estimator.predict_marginalized_over_instances(X)
            mean[:, i] = m.flatten()
            var[:, i] = v.flatten()
        return mean, var

