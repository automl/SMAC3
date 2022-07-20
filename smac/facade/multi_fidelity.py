from __future__ import annotations
from smac.chooser import Chooser

from smac.scenario import Scenario
from smac.configspace import Configuration
from smac.facade.hyperparameter import HyperparameterFacade
from smac.initial_design.random_configuration_design import RandomInitialDesign
from smac.intensification.hyperband import Hyperband

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class MultiFidelityFacade(HyperparameterFacade):
    @staticmethod
    def get_intensifier(
        scenario: Scenario, *, eta: int = 3, min_challenger=1, min_config_calls=1, max_config_calls=3
    ) -> Hyperband:
        intensifier = Hyperband(
            instances=scenario.instances,
            instance_specifics=scenario.instance_specifics,
            algorithm_walltime_limit=scenario.algorithm_walltime_limit,
            deterministic=scenario.deterministic,
            min_challenger=min_challenger,
            race_against=scenario.configspace.get_default_configuration(),
            min_config_calls=min_config_calls,
            max_config_calls=max_config_calls,
            instance_order="shuffle_once",
            eta=eta,
            seed=scenario.seed,
        )

        return intensifier

    @staticmethod
    def get_initial_design(
        scenario: Scenario,
        *,
        initial_configs: list[Configuration] | None = None,
        n_configs_per_hyperparamter: int = 10,
        max_config_ratio: float = 0.25,  # Use at most X*budget in the initial design
    ) -> RandomInitialDesign:
        return RandomInitialDesign(
            configspace=scenario.configspace,
            n_runs=scenario.n_runs,
            configs=initial_configs,
            n_configs_per_hyperparameter=n_configs_per_hyperparamter,
            max_config_ratio=max_config_ratio,
            seed=scenario.seed,
        )

    @staticmethod
    def get_configuration_chooser(scenario: Scenario) -> Chooser:
        # MultiFidelityFacade requires at least D+1 number of samples to build a model
        min_samples_model = len(scenario.configspace.get_hyperparameters()) + 1
        return Chooser(min_samples_model=min_samples_model)
