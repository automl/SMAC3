from functools import partial

from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
    InCondition,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.read_and_write import json
from ConfigSpace.util import get_one_exchange_neighbourhood

from smac.configspace.util import convert_configurations_to_array

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


get_one_exchange_neighbourhood = partial(get_one_exchange_neighbourhood, stdev=0.05, num_neighbors=8)


__all__ = [
    "ConfigurationSpace",
    "Configuration",
    "Constant",
    "CategoricalHyperparameter",
    "UniformFloatHyperparameter",
    "UniformIntegerHyperparameter",
    "InCondition",
    "json",
    "get_one_exchange_neighbourhood",
    "convert_configurations_to_array",
    "ForbiddenValueError",
]
