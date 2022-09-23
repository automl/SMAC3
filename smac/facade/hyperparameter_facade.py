from __future__ import annotations

from ConfigSpace import Configuration

from smac.acquisition.functions.expected_improvement import EI
from smac.acquisition.maximizers.local_and_random_search import (
    LocalAndSortedRandomSearch,
)
from smac.facade.abstract_facade import AbstractFacade
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.intensifier.intensifier import Intensifier
from smac.model.random_forest.random_forest import RandomForest
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.random_design.probability_design import ProbabilityRandomDesign
from smac.runhistory.encoder.log_scaled_encoder import RunHistoryLogScaledEncoder
from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class HyperparameterFacade(AbstractFacade):
    @staticmethod
    def get_model(  # type: ignore
        scenario: Scenario,
        *,
        n_trees: int = 10,
        bootstrapping: bool = True,
        ratio_features: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 2**20,
    ) -> RandomForest:
        """Returns a RandomForestWithInstances surrogate model. Please check the
        its documentation for details."""
        return RandomForest(
            log_y=True,
            n_trees=n_trees,
            bootstrapping=bootstrapping,
            ratio_features=ratio_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            configspace=scenario.configspace,
            instance_features=scenario.instance_features,
            seed=scenario.seed,
        )

    @staticmethod
    def get_acquisition_function(  # type: ignore
        scenario: Scenario,
        *,
        xi: float = 0.0,
    ) -> EI:
        """Returns an Expected Improvement acquisition function. Please check its documentation
        for details."""
        return EI(xi=xi, log=True)

    @staticmethod
    def get_acquisition_maximizer(  # type: ignore
        scenario: Scenario,
        *,
        local_search_iterations: int = 10,
        challengers: int = 10000,
    ) -> LocalAndSortedRandomSearch:
        """Local and sorted random search acquisition optimizer. Defines the optimization
        strategy for the acquisition function. Please check its documentation."""
        optimizer = LocalAndSortedRandomSearch(
            scenario.configspace,
            local_search_iterations=local_search_iterations,
            challengers=challengers,
            seed=scenario.seed,
        )

        return optimizer

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        min_challenger: int = 1,
        min_config_calls: int = 1,
        max_config_calls: int = 3,
        intensify_percentage: float = 0.5,
    ) -> Intensifier:
        """Returns an intensifier instance. Defining how to challenge the incumbent on multiple
        problem instances. Please check its documentation for details."""
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
    def get_initial_design(  # type: ignore
        scenario: Scenario,
        *,
        n_configs: int | None = None,
        n_configs_per_hyperparamter: int = 10,
        max_ratio: float = 0.1,
        additional_configs: list[Configuration] = [],
    ) -> SobolInitialDesign:
        """Returns an Sobol initial design instance. Please check its documentation for details."""
        return SobolInitialDesign(
            scenario=scenario,
            n_configs=n_configs,
            n_configs_per_hyperparameter=n_configs_per_hyperparamter,
            max_ratio=max_ratio,
            additional_configs=additional_configs,
        )

    @staticmethod
    def get_random_design(  # type: ignore
        scenario: Scenario,
        *,
        probability: float = 0.2,
    ) -> ProbabilityRandomDesign:
        """Returns a ProbabilityRandomDesign instance for interleaving BO derived configurations
        with random ones. Please check its documentation for details."""
        return ProbabilityRandomDesign(probability=probability)

    @staticmethod
    def get_multi_objective_algorithm(  # type: ignore
        scenario: Scenario,
    ) -> MeanAggregationStrategy:
        """Returns a multi-objective algorithm instance. Please check its documentation for details."""
        return MeanAggregationStrategy(scenario=scenario)

    @staticmethod
    def get_runhistory_encoder(  # type: ignore
        scenario: Scenario,
    ) -> RunHistoryLogScaledEncoder:
        return RunHistoryLogScaledEncoder(scenario)
