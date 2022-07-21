from __future__ import annotations

from smac.chooser import Chooser
from smac.configspace import Configuration
from smac.facade.hyperparameter import HyperparameterFacade
from smac.initial_design.random_configuration_design import RandomInitialDesign
from smac.intensification.hyperband import Hyperband
from smac.scenario import Scenario

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class MultiFidelityFacade(HyperparameterFacade):
    @staticmethod
    def get_intensifier(
        scenario: Scenario,
        *,
        min_challenger: int = 1,
        instance_order: str = "shuffle_once",
    ) -> Hyperband:
        return Hyperband(
            instances=scenario.instances,
            instance_specifics=scenario.instance_specifics,
            algorithm_walltime_limit=scenario.algorithm_walltime_limit,
            deterministic=scenario.deterministic,
            initial_budget=scenario.initial_budget,
            max_budget=scenario.max_budget,
            eta=scenario.eta,
            min_challenger=min_challenger,
            seed=scenario.seed,
        )

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
