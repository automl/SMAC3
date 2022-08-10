import unittest
import unittest.mock

import numpy as np
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
)

from smac.initial_design.random_design import RandomInitialDesign

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestRandomConfigurationDesign(unittest.TestCase):
    def setUp(self):
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

        self.cs = ConfigurationSpace()
        for j, get_param in enumerate(get_params):
            param_name = f"x{j}"
            self.cs.add_hyperparameter(get_param(param_name))

        for i in range(5):
            self.cs.add_hyperparameter(UniformFloatHyperparameter("x%d" % (i + len(get_params)), 0, 1))

    def test_random_configurations(self):
        kwargs = dict(
            n_runs=1000,
            configs=None,
            n_configs_per_hyperparameter=None,
            max_config_ratio=0.25,
            init_budget=1,
            seed=1,
        )
        RandomInitialDesign(configspace=self.cs, **kwargs).select_configurations()
