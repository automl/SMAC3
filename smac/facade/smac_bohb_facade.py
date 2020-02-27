import typing

from smac.facade.smac_ac_facade import SMAC4AC
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogScaledCost
from smac.optimizer.acquisition import LogEI
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.intensification.hyperband import Hyperband

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class BOHB4HPO(SMAC4AC):
    """
    Facade to use BOHB i.e., SMAC with a Hyperband intensifier for hyperparameter optimization

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

        init_kwargs = kwargs.get('initial_design_kwargs', dict())
        init_kwargs['n_configs_x_params'] = init_kwargs.get('n_configs_x_params', 10)
        init_kwargs['max_config_fracs'] = init_kwargs.get('max_config_fracs', 0.25)
        kwargs['initial_design_kwargs'] = init_kwargs

        # Intensification parameters
        # select Hyperband as the intensifier ensure respective parameters are provided
        kwargs['intensifier'] = Hyperband
        intensifier_kwargs = kwargs.get('intensifier_kwargs', dict())
        intensifier_kwargs['min_chall'] = 1
        kwargs['intensifier_kwargs'] = intensifier_kwargs
        scenario.intensification_percentage = 1e-10

        model_class = RandomForestWithInstances
        kwargs['model'] = model_class

        # == static RF settings
        model_kwargs = kwargs.get('model_kwargs', dict())
        model_kwargs['num_trees'] = model_kwargs.get('num_trees', 10)
        model_kwargs['do_bootstrapping'] = model_kwargs.get('do_bootstrapping', True)
        model_kwargs['ratio_features'] = model_kwargs.get('ratio_features', 1.0)
        model_kwargs['min_samples_split'] = model_kwargs.get('min_samples_split', 2)
        model_kwargs['min_samples_leaf'] = model_kwargs.get('min_samples_leaf', 1)
        model_kwargs['log_y'] = model_kwargs.get('log_y', True)
        kwargs['model_kwargs'] = model_kwargs

        # == Acquisition function
        kwargs['acquisition_function'] = kwargs.get('acquisition_function', LogEI)

        kwargs['runhistory2epm'] = kwargs.get('runhistory2epm', RunHistory2EPM4LogScaledCost)

        # assumes random chooser for random configs
        random_config_chooser_kwargs = kwargs.get('random_configuration_chooser_kwargs', dict())
        random_config_chooser_kwargs['prob'] = random_config_chooser_kwargs.get('prob', 0.2)
        kwargs['random_configuration_chooser_kwargs'] = random_config_chooser_kwargs

        # better improve acquisition function optimization
        # 1. increase number of sls iterations
        acquisition_function_optimizer_kwargs = kwargs.get('acquisition_function_optimizer_kwargs', dict())
        acquisition_function_optimizer_kwargs['n_sls_iterations'] = 10
        kwargs['acquisition_function_optimizer_kwargs'] = acquisition_function_optimizer_kwargs

        print(scenario.__dict__.get('always_race_against'))
        super().__init__(**kwargs)
        self.logger.info(self.__class__)

        # better improve acquisition function optimization
        # 2. more randomly sampled configurations
        self.solver.scenario.acq_opt_challengers = 10000  # type: ignore[attr-defined] # noqa F821

        # activate predict incumbent
        self.solver.epm_chooser.predict_x_best = True

        # BOHB requires at least D+1 no. of samples to build a model
        self.solver.epm_chooser.min_samples_model = len(scenario.cs.get_hyperparameters()) + 1
