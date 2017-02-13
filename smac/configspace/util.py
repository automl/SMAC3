from typing import List

import numpy as np

from smac.configspace import Configuration


def impute_inactive_hyperparameters(configs: List[Configuration]) -> np.ndarray:
    configs_array = np.array([config.get_array() for config in configs],
                             dtype=np.float64)
    configs_array[~np.isfinite(configs_array)] = -1
    return configs_array