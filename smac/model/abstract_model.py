from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, TypeVar

import copy
import warnings

import numpy as np
from ConfigSpace import ConfigurationSpace

from smac.constants import VERY_SMALL_NUMBER
from smac.model.surrogate_transformer import SurrogateTransformer
from smac.utils.configspace import get_types
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


Self = TypeVar("Self", bound="AbstractModel")


class AbstractModel:
    """Abstract implementation of the surrogate model.

    Note
    ----
    The input dimensionality of Y for training and the output dimensions of all predictions depend on the concrete
    implementation of this abstract class.

    Parameters
    ----------
    configspace : ConfigurationSpace
    instance_features : dict[str, list[int | float]] | None, defaults to None
        Features (list of int or floats) of the instances (str). The features are incorporated into the X data,
        on which the model is trained on.
    pca_components : float, defaults to 7
        Number of components to keep when using PCA to reduce dimensionality of instance features.
    seed : int
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
    ) -> None:
        self._configspace = configspace
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)
        self._instance_features = instance_features
        self._pca_components = pca_components
        self.transformer: SurrogateTransformer

        n_features = 0
        if self._instance_features is not None:
            for v in self._instance_features.values():
                if n_features == 0:
                    n_features = len(v)
                else:
                    if len(v) != n_features:
                        raise RuntimeError("Instances must have the same number of features.")

        self._n_features = n_features
        self._n_hps = len(list(self._configspace.values()))

        # Never use a lower variance than this.
        # If estimated variance < var_threshold, set to var_threshold
        self._var_threshold = VERY_SMALL_NUMBER
        self._types, self._bounds = get_types(configspace, instance_features)

        # Initial types array which is used to reset the type array at every call to `self.train()`
        self._initial_types = copy.deepcopy(self._types)

    @abstractmethod
    def build_transformer(self, normalize_y: bool = False) -> SurrogateTransformer:
        """Creates and returns the preprocessing transformer for this model.

        Parameters
        ----------
        normalize_y : bool
            Whether the transformer should normalize target values (y).

        Returns
        -------
        SurrogateTransformer
            Configured transformer for preprocessing inputs/outputs.
        """
        raise NotImplementedError

    def set_y_transform(self, func: Callable[[np.ndarray], np.ndarray] | None) -> None:
        """Sets the target transformation function used by the transformer.

        Parameters
        ----------
        func : Callable | None
            Function applied to target values before training.
        """
        self.transformer.y_transform = func

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "types": self._types,
            "bounds": self._bounds,
            "pca_components": self._pca_components,
        }

    def train(self: Self, X: np.ndarray, Y: np.ndarray) -> Self:
        """Trains the random forest on X and Y. Internally, calls the method `_train`.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameters + #features]
            Input data points.
        Y : np.ndarray [#samples, #objectives]
            The corresponding target values.

        Returns
        -------
        self : AbstractModel
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X.shape[0] ({}) != y.shape[0] ({})".format(X.shape[0], Y.shape[0]))

        X, Y = self.transformer.fit_transform(X, Y)

        if hasattr(self, "_types"):
            n_transformed_features = X.shape[1] - self._n_hps
            # For RF, adapt types list
            # if X_feats.shape[0] < self._pca, X_feats.shape[1] == X_feats.shape[0]
            self._types = np.array(
                np.hstack((self._types[: self._n_hps], np.zeros(n_transformed_features))),
                dtype=np.uint,
            )  # type: ignore

        return self._train(X, Y)

    @abstractmethod
    def _train(self: Self, X: np.ndarray, Y: np.ndarray) -> Self:
        """Trains the random forest on X and Y.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameters + #features]
            Input data points.
        Y : np.ndarray [#samples, #objectives]
            The corresponding target values.

        Returns
        -------
        self : AbstractModel
        """
        raise NotImplementedError()

    def predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predicts mean and variance for a given X. Internally, calls the method `_predict`.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameters + #features]
            Input data points.
        covariance_type: str | None, defaults to "diagonal"
            Specifies what to return along with the mean. Applied only to Gaussian Processes.
            Takes four valid inputs:
            * None: Only the mean is returned.
            * "std": Standard deviation at test points is returned.
            * "diagonal": Diagonal of the covariance matrix is returned.
            * "full": Whole covariance matrix between the test points is returned.

        Returns
        -------
        means : np.ndarray [#samples, #objectives]
            The predictive mean.
        vars : np.ndarray [#samples, #objectives] or [#samples, #samples] | None
            Predictive variance or standard deviation.
        """
        X = self.transformer.transform_X(X)
        if X.shape[1] != len(self._types):
            raise ValueError("Rows in X should have %d entries but have %d!" % (len(self._types), X.shape[1]))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Predicted variances smaller than 0. Setting those variances to 0.")
            mean, var = self._predict(X, covariance_type)

        if len(mean.shape) == 1:
            mean = mean.reshape((-1, 1))

        if var is not None and len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean, var

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predicts mean and variance for a given X.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameters + #features]
            Input data points.
        covariance_type : str | None, defaults to "diagonal"
            Specifies what to return along with the mean. Applied only to Gaussian Processes.
            Takes four valid inputs:
            * None: Only the mean is returned.
            * "std": Standard deviation at test points is returned.
            * "diagonal": Diagonal of the covariance matrix is returned.
            * "full": Whole covariance matrix between the test points is returned.

        Returns
        -------
        means : np.ndarray [#samples, #objectives]
            The predictive mean.
        vars : np.ndarray [#samples, #objectives] or [#samples, #samples] | None
            Predictive variance or standard deviation.
        """
        raise NotImplementedError()

    def predict_marginalized(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts mean and variance marginalized over all instances.

        Warning
        -------
        The input data must not include any features.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameters]
            Input data points.

        Returns
        -------
        means : np.ndarray [#samples, 1]
            The predictive mean.
        vars : np.ndarray [#samples, 1]
            The predictive variance.
        """
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != self._n_hps:
            raise ValueError(
                f"Feature mismatch: X should have {self._n_hps} hyperparameters (and no features) for this method, "
                f"but has {X.shape[1]} in total."
            )

        if self._instance_features is None:
            mean, var = self.predict(X)
            assert var is not None

            var[var < self._var_threshold] = self._var_threshold
            var[np.isnan(var)] = self._var_threshold

            return mean, var
        else:
            n_instances = len(self._instance_features)
            X_marg = self.transformer.build_marginalized_X(X)

            means, vars = self.predict(X_marg)
            assert vars is not None

            means = means.reshape(len(X), n_instances)
            vars = vars.reshape(len(X), n_instances)

            mean = np.mean(means, axis=1)
            var = np.sum(vars, axis=1) / (n_instances**2)
            var[var < self._var_threshold] = self._var_threshold

            if len(mean.shape) == 1:
                mean = mean.reshape((-1, 1))

            if len(var.shape) == 1:
                var = var.reshape((-1, 1))

            return mean, var
