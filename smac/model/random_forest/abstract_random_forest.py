from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac.model.abstract_model import AbstractModel

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class AbstractRandomForest(AbstractModel):
    """Abstract base class for all random forest models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._conditional: dict[int, bool] = dict()
        self._impute_values: dict[int, float] = dict()

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for idx, hp in enumerate(self._configspace.get_hyperparameters()):
            if idx not in self._conditional:
                parents = self._configspace.get_parents_of(hp.name)
                if len(parents) == 0:
                    self._conditional[idx] = False
                else:
                    self._conditional[idx] = True
                    if isinstance(hp, CategoricalHyperparameter):
                        self._impute_values[idx] = len(hp.choices)
                    elif isinstance(hp, OrdinalHyperparameter):
                        self._impute_values[idx] = len(hp.sequence)
                    elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                        self._impute_values[idx] = -1
                    elif isinstance(hp, Constant):
                        self._impute_values[idx] = 1
                    else:
                        raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")

            if self._conditional[idx] is True:
                nonfinite_mask = ~np.isfinite(X[:, idx])
                X[nonfinite_mask, idx] = self._impute_values[idx]

        return X
