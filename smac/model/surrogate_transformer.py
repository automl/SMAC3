from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class SurrogateTransformer:
    """
    Preprocessing pipeline for the surrogate models.

    Handles feature preprocessing (inactive imputation, scaling, PCA)
    and target preprocessing (normalization and optional transformation).

    Designed to be model-agnostic with pluggable transformation strategies.

    Parameters
    ----------
    n_hps : int
        Number of hyperparameters in input X.
    n_features: int
        Number of instance features in input X.
    impute_inactive : Callable | None, defaults to None
        Function that replaces inactive hyperparameter values.
    y_transform : Callable | None, defaults to None
        Additional transformation applied to target values before optional normalization.
    pca_components : int | None, defaults to 7
        Number of PCA components for feature reduction.
    normalize_y : bool
        Whether to standardize target values (zero mean, unit variance).
    """

    def __init__(
        self,
        n_hps: int,
        n_features: int,
        instance_features: dict[str, list[int | float]] | None,
        impute_inactive: Callable[[np.ndarray], np.ndarray] | None = None,
        y_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        pca_components: int | None = 7,
        normalize_y: bool = True,
    ):
        self._n_hps = n_hps
        self._n_features = n_features
        self._instance_features = instance_features

        self.normalize_y = normalize_y
        self.impute_inactive = impute_inactive
        self.y_transform = y_transform

        self._pca_components = pca_components
        self.pca_ = PCA(n_components=self._pca_components)
        self._pca_active = False

        self.scaler_ = MinMaxScaler()

        self.mean_y_: float | None = None
        self.std_y_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SurrogateTransformer":
        """
        Fits preprocessing steps:
        - Computes y normalization statistics
        - Fits feature scaler and PCA (if enabled)

        Parameters
        ----------
        X : np.ndarray
            Training inputss [n_samples, n_hps + n_features]
        y : np.ndarray
            Target values [n_samples, n_objectives]

        Returns
        -------
        self
        """
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != self._n_hps + self._n_features:
            raise ValueError(
                f"Feature mismatch: X should have {self._n_hps} hyperparameters + {self._n_features} features, "
                f"but has {X.shape[1]} in total."
            )

        y = y.reshape(-1)

        if self.normalize_y:
            self.mean_y_ = np.mean(y)
            self.std_y_ = np.std(y)
            if self.std_y_ == 0:
                self.std_y_ = 1.0

        if (
            self._pca_components is not None
            and X.shape[0] > self._pca_components
            and self._n_features >= self._pca_components
        ):
            X_feats = X[:, -self._n_features :]
            X_feats = self.scaler_.fit_transform(X_feats)
            X_feats = np.nan_to_num(X_feats)
            self.pca_.fit(X_feats)

            self._pca_active = True
        else:
            self._pca_active = False

        return self

    def transform_configs(self, X_cfg: np.ndarray) -> np.ndarray:
        """
        Applies configuration preprocessing.
        Handles inactive hyperparamter imputation if enabled.

        Parameters
        ----------
        X_cfg : np.ndarray
            Configuration matrix [n_samples, n_hps]

        Returns
        -------
        np.ndarray
            Transformed configurations
        """
        if X_cfg.shape[1] != self._n_hps:
            raise ValueError(f"Number of configurations should be {self._n_hps} but is {X_cfg.shape[1]}")
        X_cfg = X_cfg.copy()

        if self.impute_inactive is not None:
            X_cfg = self.impute_inactive(X_cfg)

        return X_cfg

    def transform_instance_features(self, X_inst: np.ndarray) -> np.ndarray:
        """
        Applies instance feature preprocessing.
        Applies scaling and PCA if enabled and fitted.

        Parameters
        ----------
        X_inst : np.ndarray
            Instance feature matrix [n_samples, n_features]

        Returns
        -------
        np.ndarray
            Transformed instance features
        """
        X_inst = X_inst.copy()

        if self._pca_active:
            if not hasattr(self.pca_, "components_"):
                raise RuntimeError("Transformer must be fitted before instance feature transformation.")
            X_inst = self.scaler_.transform(X_inst)
            X_inst = np.nan_to_num(X_inst)

            X_inst = self.pca_.transform(X_inst)

        return X_inst

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """
        Full feature transformation pipeline.

        Splits input into configurations and instance features,
        then applies preprocessing steps.

        Parameters
        ----------
        X : np.ndarray
            Raw input [n_samples, n_hps + n_features]

        Returns
        -------
        np.ndarray
            Fully transformed feature matrix
        """
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != self._n_hps + self._n_features:
            raise ValueError(
                f"Feature mismatch: X should have {self._n_hps} hyperparameters + {self._n_features} features, "
                f"but has {X.shape[1]} in total."
            )

        X_cfg = X[:, : self._n_hps]
        X_inst = X[:, self._n_hps :]

        X_cfg = self.transform_configs(X_cfg)
        X_inst = self.transform_instance_features(X_inst)

        return np.hstack([X_cfg, X_inst])

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """
        Applies target preprocessing:
        - Optional custom transformation
        - Mean/std normalization (if enabled)

        Parameters
        ----------
        y : np.ndarray
            target values

        Returns
        -------
        np.ndarray
            Transformed targets
        """
        if self.y_transform is not None:
            y = self.y_transform(y)

        if self.normalize_y:
            if self.mean_y_ is None or self.std_y_ is None:
                raise RuntimeError("Transformer must be fitted before calling transform_y.")
            y = (y - self.mean_y_) / self.std_y_

        return y

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies full preprocessing pipeline to X and y.

        Returns
        -------
        (X_transformed, y_transformed)
        """
        return self.transform_X(X), self.transform_y(y)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convenience method:
        fits Transformer and immediately transforms inputs.

        Returns
        -------
        (X_transformed, y_transformed)
        """
        self.fit(X, y)
        return self.transform_X(X), self.transform_y(y)

    def untransform_y(self, y: np.ndarray, var: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Reverts target preprocessing.

        Parameters
        ----------
        y : np.ndarray
            Transformed predictions
        var : np.ndarray | None, defaults to None
            Predictive variance (if available)

        Returns
        -------
        y : np.ndarray
            Original scale predictions
        var : np.ndarray, optional
            Rescaled variance (if provided)
        """
        if self.normalize_y:
            if self.mean_y_ is None or self.std_y_ is None:
                raise RuntimeError("Transformer must be fitted before calling inverse_transform_y.")
            y = y * self.std_y_ + self.mean_y_

            if var is not None:
                var = var * (self.std_y_**2)

        if var is not None:
            return y, var

        return y

    def build_marginalized_X(self, X_cfg: np.ndarray) -> np.ndarray:
        """
        Expands configuration-only input into full configuration-instance pairs
        for marginalized prediction over all available instances.

        This method performs *structural expansion only* and does NOT apply any
        feature preprocessing or transformations.

        All preprocessing steps are expected to be applied later in
        `transform_X()` inside `predict()` to ensure a single, consistent
        transformation pipeline.

        Parameters
        ----------
        X_cfg : np.ndarray
            Array of configurations with shape [n_samples, n_hyperparameters].
            Must NOT include instance features.

        Returns
        -------
        np.ndarray
            Expanded design matrix of shape
            [n_samples * n_instances, n_hyperparameters + n_features],
            where each configuration is paired with all available instance
            feature vectors in the model.
        """
        assert self._instance_features is not None
        X_inst = np.asarray(list(self._instance_features.values()))

        n_instances = X_inst.shape[0]

        X_cfg_rep = np.repeat(X_cfg, n_instances, axis=0)
        X_inst_rep = np.tile(X_inst, (X_cfg.shape[0], 1))

        return np.hstack([X_cfg_rep, X_inst_rep])
