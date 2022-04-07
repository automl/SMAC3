from typing import Dict, List, Optional, Tuple

import numpy as np

from smac.configspace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.epm.base_epm import AbstractEPM

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class BaseModel(AbstractEPM):
    def __init__(
        self,
        configspace: ConfigurationSpace,
        types: List[int],
        bounds: List[Tuple[float, float]],
        seed: int,
        instance_features: Optional[np.ndarray] = None,
        pca_components: Optional[int] = None,
    ) -> None:
        """Abstract base class for all random forest models."""
        super().__init__(
            configspace=configspace,
            types=types,
            bounds=bounds,
            seed=seed,
            instance_features=instance_features,
            pca_components=pca_components,
        )

        self.rng = np.random.RandomState(seed)
        self.conditional = dict()  # type: Dict[int, bool]
        self.impute_values = dict()  # type: Dict[int, float]

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
