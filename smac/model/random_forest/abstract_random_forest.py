from __future__ import annotations

import numpy as np

from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.model.abstract_model import AbstractModel

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class AbstractRandomForest(AbstractModel):
    def __init__(
        self,
        configspace: ConfigurationSpace,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
    ) -> None:
        """Abstract base class for all random forest models."""
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )

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
                    elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                        self._impute_values[idx] = -1
                    elif isinstance(hp, Constant):
                        self._impute_values[idx] = 1
                    else:
                        raise ValueError

            if self._conditional[idx] is True:
                nonfinite_mask = ~np.isfinite(X[:, idx])
                X[nonfinite_mask, idx] = self._impute_values[idx]

        return X

    @property
    def conditional(self) -> dict[int, bool]:
        """A dict indicating if the hyperparameter is conditioned by another hyperparameter"""
        return self._conditional

    @property
    def impute_values(self) -> dict[int, float]:
        """Values to impute missing items w.r.t. each hyperparameter"""
        return self._impute_values

