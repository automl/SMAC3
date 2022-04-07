from typing import List

import numpy as np

from smac.configspace import Configuration

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


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
    """
    return np.array([config.get_array() for config in configs], dtype=np.float64)
