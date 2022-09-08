from __future__ import annotations

import numpy as np

from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class RandomModel(AbstractModel):
    """AbstractModel which returns random values on a call to ``fit``."""

    def _train(self, X: np.ndarray, Y: np.ndarray) -> RandomModel:
        """Pseudo training on X and Y.

        Parameters
        ----------
        X : np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y : np.ndarray (N, 1)
            The corresponding target values.
        """
        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray")
        if not isinstance(Y, np.ndarray):
            raise NotImplementedError("Y has to be of type np.ndarray")

        logger.debug("(Pseudo) Fit model to data")
        return self

    def _predict(self, X: np.ndarray, cov_return_type: str | None = "diagonal_cov") -> tuple[np.ndarray, np.ndarray]:
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance features)]
        cov_return_type: Optional[str]
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

        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray")

        return self._rng.rand(len(X), 1), self._rng.rand(len(X), 1)
