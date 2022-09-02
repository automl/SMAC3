from __future__ import annotations
from functools import partial

import numpy as np

from ConfigSpace import Configuration
from ConfigSpace.util import get_one_exchange_neighbourhood

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


get_one_exchange_neighbourhood = partial(get_one_exchange_neighbourhood, stdev=0.05, num_neighbors=8)


def convert_configurations_to_array(configs: list[Configuration]) -> np.ndarray:
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
