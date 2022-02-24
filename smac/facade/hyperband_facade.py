import typing

from smac.facade.roar_facade import ROAR
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.intensification.hyperband import Hyperband

__author__ = "Ashwin Raaghav Narayanan"
__copyright__ = "Copyright 2019, ML4AAD"
__license__ = "3-clause BSD"


class HB4AC(ROAR):
    """
    Facade to use model-free Hyperband for algorithm configuration

    This facade overwrites options available via the SMAC facade.

    See Also
    --------
    :class:`~smac.facade.smac_ac_facade.SMAC4AC` for documentation of parameters.

    Attributes
    ----------
    logger
    stats : Stats
    solver : SMBO
    runhistory : RunHistory
        List with information about previous runs
    trajectory : list
        List of all incumbents

    """

    def __init__(self, **kwargs: typing.Any):
        kwargs['initial_design'] = kwargs.get('initial_design', RandomConfigurations)

        # Intensification parameters
        # select Hyperband as the intensifier ensure respective parameters are provided
        kwargs['intensifier'] = Hyperband

        # set Hyperband parameters if not given
        intensifier_kwargs = kwargs.get('intensifier_kwargs', dict())
        intensifier_kwargs['min_chall'] = 1
        if intensifier_kwargs.get('eta') is None:
            intensifier_kwargs['eta'] = 3
        if intensifier_kwargs.get('instance_order') is None:
            intensifier_kwargs['instance_order'] = 'shuffle_once'
        kwargs['intensifier_kwargs'] = intensifier_kwargs

        super().__init__(**kwargs)
        self.logger.info(self.__class__)
