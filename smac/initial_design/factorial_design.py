from __future__ import annotations

import itertools

import numpy as np
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)
from ConfigSpace.util import deactivate_inactive_hyperparameters

from smac.initial_design.abstract_initial_design import AbstractInitialDesign

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class FactorialInitialDesign(AbstractInitialDesign):
    """Factorial initial design.

    Attributes
    ----------
    configs : list[Configuration]
        List of configurations to be evaluated
        Don't pass configs to the constructor;
        otherwise factorial design is overwritten
    """

    def _select_configurations(self) -> Configuration:
        """Selects the configurations to be evaluated from the initial design.

        Returns
        -------
        config: Configuration
            initial incumbent configuration
        """
        params = self.configspace.get_hyperparameters()

        values = []
        mid = []
        for param in params:
            if isinstance(param, Constant):
                v = [param.value]
                mid.append(param.value)
            elif isinstance(param, NumericalHyperparameter):
                v = [param.lower, param.upper]
                mid.append(np.average([param.lower, param.upper]))
            elif isinstance(param, CategoricalHyperparameter):
                v = list(param.choices)
                mid.append(param.choices[0])
            elif isinstance(param, OrdinalHyperparameter):
                v = [param.sequence[0], param.sequence[-1]]
                length = len(param.sequence)
                mid.append(param.sequence[int(length / 2)])
            values.append(v)

        factorial_design = itertools.product(*values)

        configs = [self.configspace.get_default_configuration()]
        # add middle point in space
        conf_dict = dict([(p.name, v) for p, v in zip(params, mid)])
        middle_conf = deactivate_inactive_hyperparameters(conf_dict, self.configspace)
        configs.append(middle_conf)

        # add corner points
        for design in factorial_design:
            conf_dict = dict([(p.name, v) for p, v in zip(params, design)])
            conf = deactivate_inactive_hyperparameters(conf_dict, self.configspace)
            conf.origin = "Factorial Initial Design"
            configs.append(conf)

        return configs
