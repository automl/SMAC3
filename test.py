from ConfigSpace.hyperparameters import UniformFloatHyperparameter

# Import SMAC-utilities
from smac.config import Config

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.black_box import BlackBoxFacade


def quadratic(config) -> float:
    x = config["x"]

    return x * x


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    x = UniformFloatHyperparameter("x", -100, 100, default_value=-100)
    cs.add_hyperparameters([x])

    # Scenario object
    config = Config(cs, n_runs=20)
    smac = BlackBoxFacade(config, quadratic)
    smac.optimize()
