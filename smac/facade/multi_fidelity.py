from __future__ import annotations

from smac.config import Config
from smac.configspace import Configuration
from smac.facade.hyperparameter import HyperparameterFacade
from smac.initial_design.random_configuration_design import RandomInitialDesign
from smac.intensification.hyperband import Hyperband

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class MultiFidelityFacade(HyperparameterFacade):
    def _update_dependencies(self) -> None:
        super()._update_dependencies()

        # MultiFidelityFacade requires at least D+1 number of samples to build a model
        min_samples_model = len(self.config.configspace.get_hyperparameters()) + 1
        self.optimizer.epm_chooser.min_samples_model = min_samples_model

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
