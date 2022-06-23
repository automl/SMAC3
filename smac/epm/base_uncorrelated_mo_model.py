from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from smac.configspace import ConfigurationSpace
from smac.epm.base_epm import AbstractEPM

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class UncorrelatedMultiObjectiveModel(AbstractEPM):
    """Wrapper for the surrogate models to predict multiple targets.

    Only a list with the target names and the types array for the
    underlying model are mandatory. All other hyperparameters to
    model can be passed via kwargs. Consult the documentation of
    the corresponding model for the hyperparameters and their meanings.

    Parameters
    ----------
    target_names : list
        List of str, each entry is the name of one target dimension. Length
        of the list will be ``n_objectives``.
    types : List[int]
        Specifies the number of categorical values of an input dimension where
        the i-th entry corresponds to the i-th input dimension. Let's say we
        have 2 dimension where the first dimension consists of 3 different
        categorical choices and the second dimension is continuous than we
        have to pass [3, 0]. Note that we count starting from 0.
    bounds : List[Tuple[float, float]]
        bounds of input dimensions: (lower, uppper) for continuous dims; (n_cat, np.nan)
        for categorical dims
    instance_features : np.ndarray (I, K)
        Contains the K dimensional instance features of I different instances
    pca_components : float
        Number of components to keep when using PCA to reduce dimensionality of instance features.
        Requires to set n_feats (> pca_dims).
    model_kwargs: Optional[Dict[str, Any]]:
        arguments for initialing estimators

    Attributes
    ----------
    target_names:
        target names
    num_targets: int
        number of targets
    estimators: List[AbstractEPM]
        a list of estimators predicting different target values
    """

    def __init__(
        self,
        target_names: List[str],
        configspace: ConfigurationSpace,
        types: List[int],
        bounds: List[Tuple[float, float]],
        seed: int,
        instance_features: Optional[np.ndarray] = None,
        pca_components: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            configspace=configspace,
            bounds=bounds,
            types=types,
            seed=seed,
            instance_features=instance_features,
            pca_components=pca_components,
        )
        if model_kwargs is None:
            model_kwargs = {}
        self.target_names = target_names
        self.num_targets = len(self.target_names)
        self.estimators: List[AbstractEPM] = self.construct_estimators(configspace, types, bounds, model_kwargs)

    @abstractmethod
    def construct_estimators(
        self,
        configspace: ConfigurationSpace,
        types: List[int],
        bounds: List[Tuple[float, float]],
        model_kwargs: Dict[str, Any],
    ) -> List[AbstractEPM]:
        """
        Construct a list of estimators. The number of the estimators equals 'self.num_targets'
        Parameters
        ----------
        configspace : ConfigurationSpace
            Configuration space to tune for.
        types : List[int]
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass [3, 0]. Note that we count starting from 0.
        bounds : List[Tuple[float, float]]
            bounds of input dimensions: (lower, uppper) for continuous dims; (n_cat, np.nan) for categorical dims
        model_kwargs : Dict[str, Any]
            model kwargs for initializing models
        Returns
        -------
        estimators: List[AbstractEPM]
            A list of estimators
        """
        raise NotImplementedError

    def _train(self, X: np.ndarray, Y: np.ndarray) -> "UncorrelatedMultiObjectiveModel":
        """Trains the models on X and y.

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
        if len(self.estimators) == 0:
            raise ValueError("The list of estimators for this model is empty!")
        for i, estimator in enumerate(self.estimators):
            estimator.train(X, Y[:, i])

        return self

    def _predict(self, X: np.ndarray, cov_return_type: Optional[str] = "diagonal_cov") -> Tuple[np.ndarray, np.ndarray]:
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance
        features)]
        cov_return_type: typing.Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.

        Returns
        -------
        means : np.ndarray of shape = [n_samples, n_objectives]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, n_objectives]
            Predictive variance
        """
        if cov_return_type != "diagonal_cov":
            raise ValueError("'cov_return_type' can only take 'diagonal_cov' for this model")

        mean = np.zeros((X.shape[0], self.num_targets))
        var = np.zeros((X.shape[0], self.num_targets))
        for i, estimator in enumerate(self.estimators):
            m, v = estimator.predict(X)
            assert v is not None  # please mypy
            mean[:, i] = m.flatten()
            var[:, i] = v.flatten()
        return mean, var

    def predict_marginalized_over_instances(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
