from functools import partial

from ConfigSpace import ConfigurationSpace, Configuration, Constant, \
    CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, InCondition
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.read_and_write import pcs, pcs_new, json
from ConfigSpace.util import get_one_exchange_neighbourhood
from smac.configspace.util import convert_configurations_to_array

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


__all__ = ["ConfigurationSpace",
           "Configuration",
           "Constant",
           "CategoricalHyperparameter",
           "UniformFloatHyperparameter",
           "UniformIntegerHyperparameter",
           "InCondition",
           "pcs",
           "pcs_new",
           "json",
           "get_one_exchange_neighbourhood",
           "convert_configurations_to_array",
           "ForbiddenValueError"
           ]

get_one_exchange_neighbourhood = partial(get_one_exchange_neighbourhood, stdev=0.05, num_neighbors=8)
