from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from tabpfn import TabPFNRegressor

from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class TabPFNModel(AbstractModel):
    """TabPFNModel, for more details check: https://github.com/PriorLabs/TabPFN.

    Parameters
    ----------
    instance_features : dict[str, list[int | float]] | None, defaults to None
        Features (list of int or floats) of the instances (str). The features are incorporated into the X data,
        on which the model is trained on.
    pca_components : float, defaults to 7
        Number of components to keep when using PCA to reduce dimensionality of instance features.
    seed : int
    n_estimators : int, defaults to 8
        The number of estimators in the TabPFN ensemble.
    softmax_temperature : float, defaults to 0.9
        The temperature for the softmax function.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
    ) -> None:
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )

        self._tabpfn = None
        self.n_estimators = n_estimators
        self.categorical_features_indices = [
            i for i, hp in enumerate(list(configspace.values())) if isinstance(hp, CategoricalHyperparameter)
        ]
        self.softmax_temperature = softmax_temperature
        self.random_state = seed

        self._x_imputer = SimpleImputer(strategy="mean")
        self._x_pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self._x_scaler = StandardScaler()

        self._y_pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self._y_scaler = StandardScaler()

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the metadata of the model.

        Returns
        -------
            dict[str, Any]: meta data
        """
        meta = super().meta
        meta.update(
            {
                "pca_components": self._pca_components,
            }
        )
        return meta

    def _train(self, X: np.ndarray, y: np.ndarray) -> TabPFNModel:
        self._tabpfn = self._get_tabpfn()
        if self._tabpfn is None:
            raise AssertionError("TabPFNRegressor is not initialized properly!")

        # Impute, transform, scale
        X_imputed = self._x_imputer.fit_transform(X)
        X_transformed = self._x_pt.fit_transform(X_imputed)
        X_scaled = self._x_scaler.fit_transform(X_transformed)

        y = y.flatten()
        y_transformed = self._y_pt.fit_transform(y.reshape(-1, 1))
        y_scaled = self._y_scaler.fit_transform(y_transformed)
        y_scaled = y_scaled.flatten()

        self._tabpfn.fit(X_scaled, y_scaled)
        self._is_trained = True
        return self

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != len(self._types):
            raise ValueError("Rows in X should have %d entries but have %d!" % (len(self._types), X.shape[1]))

        if covariance_type != "diagonal":
            raise ValueError("`covariance_type` can only take `diagonal` for this model.")

        assert self._tabpfn is not None

        # Impute, transform, scale
        X_imputed = self._x_imputer.transform(X)
        X_transformed = self._x_pt.transform(X_imputed)
        X_scaled = self._x_scaler.transform(X_transformed)

        with torch.no_grad():
            out_dict = self._tabpfn.predict(X_scaled, output_type="full")

        # Variance estimation is difficult with TabPFN, it can have very large variances
        var = out_dict["criterion"].variance(out_dict["logits"]).cpu().detach().numpy()
        var = var.flatten()
        var = np.maximum(var, 1e-6)

        y_pred = self._y_scaler.inverse_transform(out_dict["mean"].reshape(-1, 1))
        y_pred = self._y_pt.inverse_transform(y_pred)

        return y_pred.flatten(), var

    def _get_tabpfn(self) -> TabPFNRegressor:
        """Return a TabPFNRegressor instance with the specified parameters.
        The fit_mode is set to 'low_memory' because the model is often retrained.

        Returns
        -------
            TabPFNRegressor: TabPFNRegressor.
        """
        return TabPFNRegressor(
            n_estimators=self.n_estimators,
            categorical_features_indices=self.categorical_features_indices,
            softmax_temperature=self.softmax_temperature,
            fit_mode="low_memory",
        )
