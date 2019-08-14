import numpy as np

from smac.epm.base_epm import AbstractEPM
from smac.configspace import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)


class BaseModel(AbstractEPM):

    def __init__(self, configspace, types, bounds, seed, **kwargs):
        """
        Abstract base class for all random forest models.
        """
        super().__init__(configspace=configspace, types=types, bounds=bounds, seed=seed, **kwargs)

        self.rng = np.random.RandomState(seed)
        self.impute_values = dict()

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for idx, hp in enumerate(self.configspace.get_hyperparameters()):
            if idx not in self.impute_values:
                parents = self.configspace.get_parents_of(hp.name)
                if len(parents) == 0:
                    self.impute_values[idx] = None
                else:
                    if isinstance(hp, CategoricalHyperparameter):
                        self.impute_values[idx] = len(hp.choices)
                    elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                        self.impute_values[idx] = -1
                    elif isinstance(hp, Constant):
                        self.impute_values[idx] = 1
                    else:
                        raise ValueError

            nonfinite_mask = ~np.isfinite(X[:, idx])
            X[nonfinite_mask, idx] = self.impute_values[idx]

        return X
