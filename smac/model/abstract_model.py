from __future__ import annotations

from abc import abstractmethod
from typing import Any, TypeVar

import copy
import warnings

import numpy as np
from ConfigSpace import ConfigurationSpace
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from smac.constants import VERY_SMALL_NUMBER
from smac.utils.configspace import get_types
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
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

        n_features = 0
        if self._instance_features is not None:
            for v in self._instance_features.values():
                if n_features == 0:
                    n_features = len(v)
                else:
                    if len(v) != n_features:
                        raise RuntimeError("Instances must have the same number of features.")

        self._n_features = n_features
        self._n_hps = len(self._configspace.get_hyperparameters())

        self._pca = PCA(n_components=self._pca_components)
        self._scaler = MinMaxScaler()
        self._apply_pca = False

        # Never use a lower variance than this.
        # If estimated variance < var_threshold, set to var_threshold
        self._var_threshold = VERY_SMALL_NUMBER
        self._types, self._bounds = get_types(configspace, instance_features)

        # Initial types array which is used to reset the type array at every call to `self.train()`
        self._initial_types = copy.deepcopy(self._types)

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
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != self._n_hps + self._n_features:
            raise ValueError(
                f"Feature mismatch: X should have {self._n_hps} hyperparameters + {self._n_features} features, "
                f"but has {X.shape[1]} in total."
            )

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X.shape[0] ({}) != y.shape[0] ({})".format(X.shape[0], Y.shape[0]))

        # Reduce dimensionality of features if larger than PCA_DIM
        if (
            self._pca_components is not None
            and X.shape[0] > self._pca.n_components
            and self._n_features >= self._pca_components
        ):
            X_feats = X[:, -self._n_features :]

            # Scale features
            X_feats = self._scaler.fit_transform(X_feats)
            X_feats = np.nan_to_num(X_feats)  # if features with max == min

            # PCA
            X_feats = self._pca.fit_transform(X_feats)
            X = np.hstack((X[:, : self._n_hps], X_feats))

            if hasattr(self, "_types"):
                # For RF, adapt types list
                # if X_feats.shape[0] < self._pca, X_feats.shape[1] == X_feats.shape[0]
                self._types = np.array(
                    np.hstack((self._types[: self._n_hps], np.zeros(X_feats.shape[1]))),
                    dtype=np.uint,
                )  # type: ignore

            self._apply_pca = True
        else:
            self._apply_pca = False

            if hasattr(self, "_types"):
                self._types = copy.deepcopy(self._initial_types)

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
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != self._n_hps + self._n_features:
            raise ValueError(
                f"Feature mismatch: X should have {self._n_hps} hyperparameters + {self._n_features} features, "
                f"but has {X.shape[1]} in total."
            )

        if self._apply_pca:
            try:
                X_feats = X[:, -self._n_features :]
                X_feats = self._scaler.transform(X_feats)
                X_feats = self._pca.transform(X_feats)
                X = np.hstack((X[:, : self._n_hps], X_feats))
            except NotFittedError:
                # PCA not fitted if only one training sample
                pass

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

            mean = np.zeros(X.shape[0])
            var = np.zeros(X.shape[0])
            for i, x in enumerate(X):
                features = np.array(list(self._instance_features.values()))
                x_tiled = np.tile(x, (n_instances, 1))
                X_ = np.hstack((x_tiled, features))

                means, vars = self.predict(X_)
                assert vars is not None

                # VAR[1/n (X_1 + ... + X_n)] =
                # 1/n^2 * ( VAR(X_1) + ... + VAR(X_n))
                # for independent X_1 ... X_n
                var_x = np.sum(vars) / (len(vars) ** 2)
                if var_x < self._var_threshold:
                    var_x = self._var_threshold

                var[i] = var_x
                mean[i] = np.mean(means)

            if len(mean.shape) == 1:
                mean = mean.reshape((-1, 1))

            if len(var.shape) == 1:
                var = var.reshape((-1, 1))

            return mean, var
