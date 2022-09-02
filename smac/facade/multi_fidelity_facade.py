from __future__ import annotations

from ConfigSpace import Configuration
from smac.facade.hyperparameter_facade import HyperparameterFacade
from smac.initial_design.random_design import RandomInitialDesign
from smac.intensification.hyperband import Hyperband
from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class MultiFidelityFacade(HyperparameterFacade):
    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        eta: int = 3,
        min_challenger: int = 1,
        intensify_percentage: float = 0.5,
        n_seeds: int = 1,
    ) -> Hyperband:
        """Returns a Hyperband intensifier instance. Please check its documentation for details."""
        return Hyperband(
            scenario=scenario,
            eta=eta,
            min_challenger=min_challenger,
            intensify_percentage=intensify_percentage,
            n_seeds=n_seeds,
        )

    @staticmethod
    def get_initial_design(  # type: ignore
        scenario: Scenario,
        *,
        configs: list[Configuration] | None = None,
        n_configs: int | None = None,
        n_configs_per_hyperparamter: int = 10,
        max_config_ratio: float = 0.25,  # Use at most X*budget in the initial design
    ) -> RandomInitialDesign:
        """Returns a random initial design instance. Please check its documentation for details."""
        return RandomInitialDesign(
            scenario=scenario,
            configs=configs,
            n_configs=n_configs,
            n_configs_per_hyperparameter=n_configs_per_hyperparamter,
            max_config_ratio=max_config_ratio,
        )
