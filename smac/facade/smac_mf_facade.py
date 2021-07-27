import typing

from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogScaledCost
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.intensification.hyperband import Hyperband

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class SMAC4MF(SMAC4HPO):
    """
    Facade to use SMAC with a Hyperband intensifier for hyperparameter optimization using multiple
    fidelities

    see smac.facade.smac_Facade for API
    This facade overwrites options available via the SMAC facade

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
        """
        Constructor
        see ~smac.facade.smac_facade for documentation
        """

        scenario = kwargs['scenario']

        kwargs['initial_design'] = kwargs.get('initial_design', RandomConfigurations)
        kwargs['runhistory2epm'] = kwargs.get('runhistory2epm', RunHistory2EPM4LogScaledCost)

        # Intensification parameters
        # select Hyperband as the intensifier ensure respective parameters are provided
        if kwargs.get('intensifier') is None:
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

        # better improve acquisition function optimization
        # 2. more randomly sampled configurations
        self.solver.scenario.acq_opt_challengers = 10000  # type: ignore[attr-defined] # noqa F821

        # activate predict incumbent
        self.solver.epm_chooser.predict_x_best = True

        # SMAC4MF requires at least D+1 no. of samples to build a model
        self.solver.epm_chooser.min_samples_model = len(scenario.cs.get_hyperparameters()) + 1
