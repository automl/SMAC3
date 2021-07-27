import typing

from scipy.stats.qmc import Sobol

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant

from smac.initial_design.initial_design import InitialDesign

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class SobolDesign(InitialDesign):
    """ Sobol sequence design with a scrambled Sobol sequence.

    See https://scipy.github.io/devdocs/reference/generated/scipy.stats.qmc.Sobol.html for further information

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

        dim = len(params) - constants
        sobol_gen = Sobol(d=dim, scramble=True, seed=self.rng.randint(low=0, high=10000000))
        sobol = sobol_gen.random(self.init_budget)

        return self._transform_continuous_designs(design=sobol,
                                                  origin='Sobol',
                                                  cs=self.cs)
