from typing import List

import numpy as np

from smac.configspace import Configuration, ConfigurationSpace


def convert_configurations_to_array(configs: List[Configuration]) -> np.ndarray:
    """Impute inactive hyperparameters in configurations with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configs : List[Configuration]
        List of configuration objects.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    configs_array = np.array([config.get_array() for config in configs],
                             dtype=np.float64)
    configuration_space = configs[0].configuration_space
    return impute_inactive_values(configuration_space, configs_array)


def impute_inactive_values(
        configuration_space: ConfigurationSpace,
        configs_array: np.ndarray
) -> np.ndarray:
    """Impute inactive hyperparameters in configuration array with a value outside their default range.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configuration_space : ConfigurationSpace

    configs_array : np.ndarray
        Array of configurations.

    Returns
    -------
    np.ndarray
    """
    for hp in configuration_space.get_hyperparameters():
        idx = configuration_space.get_idx_by_hyperparameter_name(hp.name)
        nonfinite_mask = ~np.isfinite(configs_array[:, idx])
        configs_array[nonfinite_mask, idx] = -1

    return configs_array
