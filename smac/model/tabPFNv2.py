from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
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

    def _train(self, X: np.ndarray, y: np.ndarray):
        self._tabpfn = self._get_tabpfn()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self._tabpfn.fit(X, y)
        self._is_trained = True
        return self
        
    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2d array, got {len(X.shape)}d array!")

        if X.shape[1] != len(self._types):
            raise ValueError(
                f"Rows in X should have {len(self._types)} entries but have {X.shape[1]}!"
            )

        if covariance_type != "diagonal":
            raise ValueError("`covariance_type` can only take `diagonal` for this model.")

        if self._tabpfn is None or not self._is_trained:
            raise RuntimeError("TabPFNModel was asked to predict before being trained.")

        X = np.asarray(X, dtype=np.float32)

        out = self._tabpfn.predict(X, output_type="full")

        mean = np.asarray(out["mean"], dtype=np.float64).reshape(-1, 1)

        var = out["criterion"].variance(out["logits"])
        var = np.asarray(var.detach().cpu(), dtype=np.float64).reshape(-1, 1)
        var = np.maximum(var, 1e-12)

        return mean, var

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
            model_path=''
        )
