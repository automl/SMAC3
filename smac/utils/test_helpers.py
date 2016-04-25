from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter


def get_branin_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter('x', -5, 10))
    cs.add_hyperparameter(UniformFloatHyperparameter('y', 0, 15))
    return cs