from __future__ import annotations

from ConfigSpace import Configuration

from smac.acquisition.function.expected_improvement import EI
from smac.acquisition.maximizer.local_and_random_search import (
    LocalAndSortedRandomSearch,
)
from smac.facade.abstract_facade import AbstractFacade
from smac.initial_design.default_design import DefaultInitialDesign
from smac.intensifier.intensifier import Intensifier
from smac.model.random_forest.random_forest import RandomForest
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.random_design.probability_design import ProbabilityRandomDesign
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class AlgorithmConfigurationFacade(AbstractFacade):
    @staticmethod
    def get_model(  # type: ignore
        scenario: Scenario,
        *,
        n_trees: int = 10,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 20,
        bootstrapping: bool = True,
        pca_components: int = 4,
    ) -> RandomForest:
        """Returns a random forest as surrogate model.

        Parameters
        ----------
        n_trees : int, defaults to 10
            The number of trees in the random forest.
        ratio_features : float, defaults to 5.0 / 6.0
            The ratio of features that are considered for splitting.
        min_samples_split : int, defaults to 3
            The minimum number of data points to perform a split.
        min_samples_leaf : int, defaults to 3
            The minimum number of data points in a leaf.
        max_depth : int, defaults to 20
            The maximum depth of a single tree.
        bootstrapping : bool, defaults to True
            Enables bootstrapping.
        pca_components : float, defaults to 4
            Number of components to keep when using PCA to reduce dimensionality of instance features.
        """
        return RandomForest(
            configspace=scenario.configspace,
            n_trees=n_trees,
            ratio_features=ratio_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            bootstrapping=bootstrapping,
            log_y=False,
            instance_features=scenario.instance_features,
            pca_components=pca_components,
            seed=scenario.seed,
        )

    @staticmethod
    def get_acquisition_function(  # type: ignore
        scenario: Scenario,
        *,
        xi: float = 0.0,
    ) -> EI:
        """Returns an Expected Improvement acquisition function.

        Parameters
        ----------
        scenario : Scenario
        xi : float, defaults to 0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        return EI(xi=xi)

    @staticmethod
    def get_acquisition_maximizer(  # type: ignore
        scenario: Scenario,
    ) -> LocalAndSortedRandomSearch:
        """Returns local and sorted random search as acquisition maximizer."""
        optimizer = LocalAndSortedRandomSearch(
            scenario.configspace,
            seed=scenario.seed,
        )

        return optimizer

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        min_config_calls=1,
        max_config_calls=2000,
        min_challenger=1,
        intensify_percentage: float = 0.5,
    ) -> Intensifier:
        """Returns ``Intensifier`` as intensifier. Uses the default configuration for ``race_against``.

        Parameters
        ----------
        scenario : Scenario
        min_config_calls : int, defaults to 1
            Minimum number of trials per config (summed over all calls to intensify).
        max_config_calls : int, defaults to 2000
            Maximum number of trials per config (summed over all calls to intensify).
        min_challenger : int, defaults to 2
            Minimal number of challengers to be considered (even if time_bound is exhausted earlier).
        intensify_percentage : float, defaults to 0.5
            How much percentage of the time should configurations be intensified (evaluated on higher budgets or
            more instances). This parameter is accessed in the SMBO class.
        """
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
        additional_configs: list[Configuration] = [],
    ) -> DefaultInitialDesign:
        """Returns an initial design, which returns the default configuration.

        Parameters
        ----------
        additional_configs: list[Configuration], defaults to []
            Adds additional configurations to the initial design.
        """
        return DefaultInitialDesign(
            scenario=scenario,
            additional_configs=additional_configs,
        )

    @staticmethod
    def get_random_design(  # type: ignore
        scenario: Scenario,
        *,
        probability: float = 0.5,
    ) -> ProbabilityRandomDesign:
        """Returns ``ProbabilityRandomDesign`` for interleaving configurations.

        Parameters
        ----------
        probability : float, defaults to 0.5
            Probability that a configuration will be drawn at random.
        """
        return ProbabilityRandomDesign(probability=probability, seed=scenario.seed)

    @staticmethod
    def get_multi_objective_algorithm(  # type: ignore
        scenario: Scenario,
        *,
        objective_weights: list[float] | None = None,
    ) -> MeanAggregationStrategy:
        """Returns the mean aggregation strategy for the multi objective algorithm.

        Parameters
        ----------
        scenario : Scenario
        objective_weights : list[float] | None, defaults to None
            Weights for an weighted average. Must be of the same length as the number of objectives.
        """
        return MeanAggregationStrategy(
            scenario=scenario,
            objective_weights=objective_weights,
        )

    @staticmethod
    def get_runhistory_encoder(scenario: Scenario) -> RunHistoryEncoder:
        """Returns the default runhistory encoder."""
        return RunHistoryEncoder(scenario)
