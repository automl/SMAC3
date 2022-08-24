from __future__ import annotations

from smac.acquisition.functions.expected_improvement import EI
from smac.acquisition.local_and_random_search import LocalAndSortedRandomSearch
from smac.random_design.probability_design import ProbabilityRandomDesign
from smac.configspace import Configuration
from smac.facade.facade import Facade
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.intensification.intensification import Intensifier
from smac.model.random_forest.random_forest_with_instances import (
    RandomForestWithInstances,
)
from smac.model.utils import get_types
from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.runhistory.encoder.log_scaled_encoder import RunHistoryLogScaledEncoder
from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class HyperparameterFacade(Facade):
    @staticmethod
    def get_model(
        scenario: Scenario,
        *,
        n_trees: int = 10,
        bootstrapping: bool = True,
        ratio_features: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 2**20,
    ) -> RandomForestWithInstances:
        types, bounds = get_types(scenario.configspace)

        return RandomForestWithInstances(
            types=types,
            bounds=bounds,
            log_y=True,
            num_trees=n_trees,
            do_bootstrapping=bootstrapping,
            ratio_features=ratio_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            configspace=scenario.configspace,
            instance_features=scenario.instance_features,
            seed=scenario.seed,
        )

    @staticmethod
    def get_acquisition_function(scenario: Scenario, *, xi: float = 0.0) -> EI:
        return EI(xi=xi, log=True)

    @staticmethod
    def get_acquisition_optimizer(
        scenario: Scenario,
        *,
        local_search_iterations: int = 10,
        challengers: int = 10000,
    ) -> LocalAndSortedRandomSearch:
        optimizer = LocalAndSortedRandomSearch(
            scenario.configspace,
            local_search_iterations=local_search_iterations,
            challengers=challengers,
            seed=scenario.seed,
        )

        return optimizer

    @staticmethod
    def get_intensifier(
        scenario: Scenario,
        *,
        min_challenger: int = 1,
        min_config_calls: int = 1,
        max_config_calls: int = 3,
        intensify_percentage: float = 0.5,
    ) -> Intensifier:
        intensifier = Intensifier(
            scenario=scenario,
            min_challenger=min_challenger,
            race_against=scenario.configspace.get_default_configuration(),
            min_config_calls=min_config_calls,
            max_config_calls=max_config_calls,
            intensify_percentage=intensify_percentage,
        )

        return intensifier

    @staticmethod
    def get_initial_design(
        scenario: Scenario,
        *,
        configs: list[Configuration] | None = None,
        n_configs: int | None = None,
        n_configs_per_hyperparamter: int = 10,
        max_config_ratio: float = 0.25,  # Use at most X*budget in the initial design
    ) -> SobolInitialDesign:
        return SobolInitialDesign(
            scenario=scenario,
            configs=configs,
            n_configs=n_configs,
            n_configs_per_hyperparameter=n_configs_per_hyperparamter,
            max_config_ratio=max_config_ratio,
        )

    @staticmethod
    def get_random_design(
        scenario: Scenario,
        *,
        probability: float = 0.2,
    ) -> ProbabilityRandomDesign:
        return ProbabilityRandomDesign(probability=probability)

    @staticmethod
    def get_multi_objective_algorithm(scenario: Scenario) -> AbstractMultiObjectiveAlgorithm:
        return MeanAggregationStrategy(scenario.seed)

    @staticmethod
    def get_runhistory_encoder(scenario: Scenario) -> RunHistoryLogScaledEncoder:
        return RunHistoryLogScaledEncoder(scenario)
