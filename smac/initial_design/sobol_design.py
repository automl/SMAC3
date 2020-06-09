import typing

from scipy.optimize._shgo_lib.sobol_seq import Sobol

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant

from smac.initial_design.initial_design import InitialDesign

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class SobolDesign(InitialDesign):
    """ Sobol sequence design

    Attributes
    ----------
    configs : typing.List[Configuration]
        List of configurations to be evaluated
        Don't pass configs to the constructor;
        otherwise factorial design is overwritten
    """

    def _select_configurations(self) -> typing.List[Configuration]:
        """Selects a single configuration to run

        Returns
        -------
        config: Configuration
            initial incumbent configuration
        """

        params = self.cs.get_hyperparameters()

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        sobol_gen = Sobol()
        sobol = sobol_gen.i4_sobol_generate(len(params) - constants, self.init_budget)

        return self._transform_continuous_designs(design=sobol,
                                                  origin='Sobol',
                                                  cs=self.cs)
