from typing import List

import numpy as np

from smac.configspace import Configuration


def impute_inactive_hyperparameters(configs: List[Configuration]) -> np.ndarray:
    configs_array = np.array([config.get_array() for config in configs],
                             dtype=np.float64)
    configuration_space = configs[0].configuration_space
    for hp in configuration_space.get_hyperparameters():
        default = hp._inverse_transform(hp.default)
        idx = configuration_space.get_idx_by_hyperparameter_name(hp.name)

        # Create a mask which is True for all non-finite entries in column idx!
        column_mask = np.zeros(configs_array.shape, dtype=np.bool)
        column_mask[:, idx] = True
        nonfinite_mask = ~np.isfinite(configs_array)
        mask = column_mask & nonfinite_mask

        configs_array[mask] = default
    return configs_array