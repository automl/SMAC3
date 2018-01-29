from typing import List

import numpy as np

from smac.configspace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter


class ConfigurationConverter(object):
    def __init__(self):
        self.impute_values = dict()

    def __call__(self, configs: List[Configuration]) -> np.ndarray:
        return self.convert_configurations_to_array(configs)

    def convert_configurations_to_array(
            self,
            configs: List[Configuration]
    ) -> np.ndarray:
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
        return self.impute_default_values(configuration_space, configs_array)


    def impute_default_values(
            self,
            configuration_space: ConfigurationSpace,
            configs_array: np.ndarray
    ) -> np.ndarray:
        """Impute inactive hyperparameters in configuration array with -1.

        Necessary to apply an EPM to the data.

        Parameters
        ----------
        configuration_space : ConfigurationSpace

        configs_array : np.ndarray
            Array of configurations.

        Returns
        -------
        np.ndarray
            Array with configuration hyperparameters. Inactive values are imputed
            with their default value.
        """
        for idx, hp in enumerate(configuration_space.get_hyperparameters()):
            if idx not in self.impute_values:
                parents = configuration_space.get_parents_of(hp.name)
                if len(parents) == 0:
                    self.impute_values[idx] = None
                else:
                    if isinstance(hp, CategoricalHyperparameter):
                        self.impute_values[idx] = len(hp.choices)
                    elif isinstance(hp, (UniformFloatHyperparameter,
                                         UniformIntegerHyperparameter)):
                        self.impute_values[idx] = -1
                    elif isinstance(hp, Constant):
                        self.impute_values[idx] = 1
                    else:
                        raise ValueError

            nonfinite_mask = ~np.isfinite(configs_array[:, idx])
            configs_array[nonfinite_mask, idx] = self.impute_values[idx]

        return configs_array


# This keeps the syntax equal to the previous version in which
# convert_configuration_to_array was a function - however, using a class has
# the advantage of not having to look up the imputation value in each iteration.
convert_configurations_to_array = ConfigurationConverter()
