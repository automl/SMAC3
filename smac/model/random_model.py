from __future__ import annotations

from typing import Any

import numpy as np

from smac.model.abstract_model import AbstractModel
from smac.model.surrogate_transformer import SurrogateTransformer
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class RandomModel(AbstractModel):
    """AbstractModel which returns random values on a call to `fit`."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.transformer = self.build_transformer()

    def build_transformer(self, normalize_y: bool = False) -> SurrogateTransformer:  # noqa: D102
        return SurrogateTransformer(
            n_hps=self._n_hps,
            n_features=self._n_features,
            instance_features=self._instance_features,
            impute_inactive=None,
            normalize_y=normalize_y,
            pca_components=None,
        )

    def _train(self, X: np.ndarray, Y: np.ndarray) -> RandomModel:
        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray.")
        if not isinstance(Y, np.ndarray):
            raise NotImplementedError("Y has to be of type np.ndarray.")

        logger.debug("(Pseudo) fit model to data.")
        return self

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if covariance_type != "diagonal":
            raise ValueError("`covariance_type` can only take `diagonal` for this model.")

        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray.")

        return self._rng.rand(len(X), 1), self._rng.rand(len(X), 1)
