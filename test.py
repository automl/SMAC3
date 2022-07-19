from ConfigSpace.hyperparameters import UniformFloatHyperparameter

# Import SMAC-utilities
from smac import Config

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac import BlackBoxFacade, HyperparameterFacade, MultiFidelityFacade


def quadratic(config) -> float:
    x = config["x"]

    return x * x


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    x = UniformFloatHyperparameter("x", -100, 100, default_value=-100)
    cs.add_hyperparameters([x])

    # Scenario object
    config = Config(cs, name="hey", n_runs=60)
    # smac = BlackBoxFacade(config, quadratic)
    smac = HyperparameterFacade(config, quadratic)

    smac.optimize()
