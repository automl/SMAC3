from __future__ import annotations

import numpy as np

from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class RandomModel(AbstractModel):
    """AbstractModel which returns random values on a call to `fit`."""

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
