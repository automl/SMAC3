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

        self.conditional: dict[int, bool] = dict()
        self.impute_values: dict[int, float] = dict()

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for idx, hp in enumerate(self.configspace.get_hyperparameters()):
            if idx not in self.conditional:
                parents = self.configspace.get_parents_of(hp.name)
                if len(parents) == 0:
                    self.conditional[idx] = False
                else:
                    self.conditional[idx] = True
                    if isinstance(hp, CategoricalHyperparameter):
                        self.impute_values[idx] = len(hp.choices)
                    elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                        self.impute_values[idx] = -1
                    elif isinstance(hp, Constant):
                        self.impute_values[idx] = 1
                    else:
                        raise ValueError

            if self.conditional[idx] is True:
                nonfinite_mask = ~np.isfinite(X[:, idx])
                X[nonfinite_mask, idx] = self.impute_values[idx]

        return X
