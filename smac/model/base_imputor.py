from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class BaseImputor:
    """Abstract implementation of the Imputation API."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def impute(
        self,
        censored_X: np.ndarray,
        censored_y: np.ndarray,
        uncensored_X: np.ndarray,
        uncensored_y: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Imputes censored runs and returns new y values.

        Parameters
        ----------
        censored_X : np.ndarray [N, M]
            Feature array of all censored runs.
        censored_y : np.ndarray [N, 1]
            Target values for all runs censored runs.
        uncensored_X : np.ndarray [N, M]
            Feature array of all non-censored runs.
        uncensored_y : np.ndarray [N, 1]
            Target values for all non-censored runs.

        Returns
        -------
        imputed_y: np.ndarray
            Same shape as censored_y [N, 1]
        """