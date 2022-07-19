from random import Random
from typing import Any
from smac.config import Config

from smac.facade.hyperparameter_optimization import SMAC4HPO, HyperparameterOptimizationFacade
from smac.initial_design.random_configuration_design import RandomInitialDesign
from smac.intensification.hyperband import Hyperband
from smac.runhistory.runhistory_transformer import RunhistoryLogScaledTransformer

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class MultiFidelityFacade(HyperparameterOptimizationFacade):
    @staticmethod
    def get_intensifier(
        config: Config, *, eta: int = 3, min_challenger=1, min_config_calls=1, max_config_calls=3
    ) -> Hyperband:
        intensifier = Hyperband(
            instances=config.instances,
            instance_specifics=config.instance_specifics,
            algorithm_walltime_limit=config.algorithm_walltime_limit,
            deterministic=config.deterministic,
            min_challenger=min_challenger,
            race_against=config.configspace.get_default_configuration(),
            min_config_calls=min_config_calls,
            max_config_calls=max_config_calls,
            instance_order="shuffle_once",
            eta=eta,
            seed=config.seed,
        )

        return intensifier

    @staticmethod
    def get_initial_design(
        config: Config,
        *,
        initial_configs: list[Configuration] | None = None,
        n_configs_per_hyperparamter: int = 10,
        max_config_ratio: float = 0.25,  # Use at most X*budget in the initial design
    ) -> RandomInitialDesign:
        return RandomInitialDesign(
            configspace=config.configspace,
            n_runs=config.n_runs,
            configs=initial_configs,
            n_configs_per_hyperparameter=n_configs_per_hyperparamter,
            max_config_ratio=max_config_ratio,
            seed=config.seed,
        )


class SMAC4MF(SMAC4HPO):
    """Facade to use SMAC with a Hyperband intensifier for hyperparameter optimization using
    multiple fidelities.

    see smac.facade.smac_Facade for API
    This facade overwrites options available via the SMAC facade

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

    def __init__(self, **kwargs: Any):
        scenario = kwargs["scenario"]

        kwargs["initial_design"] = kwargs.get("initial_design", RandomInitialDesign)
        kwargs["runhistory2epm"] = kwargs.get("runhistory2epm", RunhistoryLogScaledTransformer)

        # Intensification parameters
        # select Hyperband as the intensifier ensure respective parameters are provided
        if kwargs.get("intensifier") is None:
            kwargs["intensifier"] = Hyperband

        # set Hyperband parameters if not given
        intensifier_kwargs = kwargs.get("intensifier_kwargs", dict())
        intensifier_kwargs["min_chall"] = 1
        if intensifier_kwargs.get("eta") is None:
            intensifier_kwargs["eta"] = 3
        if intensifier_kwargs.get("instance_order") is None:
            intensifier_kwargs["instance_order"] = "shuffle_once"
        kwargs["intensifier_kwargs"] = intensifier_kwargs

        super().__init__(**kwargs)
        self.logger.info(self.__class__)

        # better improve acquisition function optimization
        # 2. more randomly sampled configurations
        self.solver.scenario.acq_opt_challengers = 10000  # type: ignore[attr-defined] # noqa F821

        # activate predict incumbent
        self.solver.epm_chooser.predict_x_best = True

        # SMAC4MF requires at least D+1 no. of samples to build a model
        self.solver.epm_chooser.min_samples_model = len(scenario.cs.get_hyperparameters()) + 1
