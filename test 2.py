from ConfigSpace.hyperparameters import UniformFloatHyperparameter

# Import SMAC-utilities
from smac.config import Config

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.algorithm_configuration_facade import AlgorithmConfigurationFacade


def rosenbrock_2d(x):
    x1 = x["x0"]
    x2 = x["x1"]

    val = 100.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
    return val


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])

    # Scenario object
    config = Config(cs)
    smac = AlgorithmConfigurationFacade(config, rosenbrock_2d)

    smac.optimize()
