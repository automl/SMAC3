import unittest
import unittest.mock

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter,\
    Constant, CategoricalHyperparameter, OrdinalHyperparameter

from smac.initial_design.factorial_design import FactorialInitialDesign

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestFactorial(unittest.TestCase):
    def test_factorial(self):
        def get_uniform_param(name: str):
            return UniformFloatHyperparameter(name, 0, 1)

        def get_constant_param(name: str):
            return Constant(name, 0.)

        def get_categorical_param(name: str):
            return CategoricalHyperparameter(name, choices=["a", "b", "c"])

        def get_ordinal_param(name: str):
            return OrdinalHyperparameter(name, [8, 6, 4, 2])

        get_params = [
            get_uniform_param,
            get_constant_param,
            get_categorical_param,
            get_ordinal_param
        ]

        dims = np.arange(1, 5)
        for n_dim in dims:
            cs = ConfigurationSpace()
            for i in range(n_dim):
                for j, get_param in enumerate(get_params):
                    param_name = f"x{i+1}_{j}"
                    cs.add_hyperparameter(get_param(param_name))

            factorial_kwargs = dict(
                rng=np.random.RandomState(1),
                traj_logger=unittest.mock.Mock(),
                ta_run_limit=1000,
                configs=None,
                n_configs_x_params=None,
                max_config_fracs=0.25,
                init_budget=1,
            )
            FactorialInitialDesign(
                cs=cs,
                **factorial_kwargs
            ).select_configurations()
