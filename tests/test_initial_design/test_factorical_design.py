import numpy as np
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
)

from smac.initial_design.factorial_design import FactorialInitialDesign

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def test_factorial(make_scenario):
    def get_uniform_param(name: str):
        return UniformFloatHyperparameter(name, 0, 1)

    def get_constant_param(name: str):
        return Constant(name, 0.0)

    def get_categorical_param(name: str):
        return CategoricalHyperparameter(name, choices=["a", "b", "c"])

    def get_ordinal_param(name: str):
        return OrdinalHyperparameter(name, [8, 6, 4, 2])

    get_params = [
        get_uniform_param,
        get_constant_param,
        get_categorical_param,
        get_ordinal_param,
    ]

    dims = np.arange(1, 5)
    for n_dim in dims:
        cs = ConfigurationSpace()
        for i in range(n_dim):
            for j, get_param in enumerate(get_params):
                param_name = f"x{i+1}_{j}"
                cs.add_hyperparameter(get_param(param_name))

        design = FactorialInitialDesign(
            make_scenario(configspace=cs),
            n_configs=10,
            n_configs_per_hyperparameter=None,
            max_ratio=0.25,
            seed=1,
        )
        design.select_configurations()
