from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_branin_config_space() -> ConfigurationSpace:
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter('x', -5, 10))
    cs.add_hyperparameter(UniformFloatHyperparameter('y', 0, 15))
    return cs
