import unittest
import unittest.mock

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter,\
    Constant, CategoricalHyperparameter, OrdinalHyperparameter, ForbiddenEqualsClause

from smac.initial_design.latin_hypercube_design import LHDesign

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestLHDesign(unittest.TestCase):
    def setUp(self):
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

        self.cs = ConfigurationSpace()
        for j, get_param in enumerate(get_params):
            param_name = f"x{j}"
            self.cs.add_hyperparameter(get_param(param_name))

        param_constrained = CategoricalHyperparameter("constrained", choices=["a", "b", "c"])
        self.cs.add_hyperparameter(param_constrained)
        self.cs.add_forbidden_clause(ForbiddenEqualsClause(param_constrained, "b"))

        for i in range(5):
            self.cs.add_hyperparameter(UniformFloatHyperparameter('x%d' % (i + len(get_params)), 0, 1))

    def test_latin_hypercube_design(self):
        kwargs = dict(
            rng=np.random.RandomState(1),
            traj_logger=unittest.mock.Mock(),
            ta_run_limit=1000,
            configs=None,
            n_configs_x_params=None,
            max_config_fracs=0.25,
            init_budget=1000,
        )
        LHDesign(
            cs=self.cs,
            **kwargs
        ).select_configurations()
