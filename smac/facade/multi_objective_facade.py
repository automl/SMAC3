from __future__ import annotations

from ConfigSpace import Configuration

from smac.acquisition.function.expected_hypervolume import EHVI, PHVI
from smac.acquisition.maximizer.multi_objective_search import (
    MOLocalAndSortedRandomSearch,
)
from smac.facade.abstract_facade import AbstractFacade
from smac.initial_design.default_design import DefaultInitialDesign
from smac.intensifier.intensifier import Intensifier
from smac.intensifier.mixins import intermediate_decision, intermediate_update
from smac.intensifier.multi_objective_intensifier import MOIntensifier
from smac.model.multi_objective_model import MultiObjectiveModel
from smac.model.random_forest.random_forest import RandomForest
from smac.multi_objective.aggregation_strategy import NoAggregationStrategy
from smac.random_design.probability_design import ProbabilityRandomDesign
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class MultiObjectiveFacade(AbstractFacade):
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
        models = []
        for objective in scenario.objectives:
            models.append(
                RandomForest(
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
            )

        return MultiObjectiveModel(models=models, objectives=scenario.objectives)

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        max_config_calls: int = 2000,
        max_incumbents: int = 10,
    ) -> Intensifier:
        """Returns ``MOIntensifier`` as intensifier. Uses the default configuration for ``race_against``.

        Parameters
        ----------
        scenario : Scenario
        max_config_calls : int, defaults to 2000
            Maximum number of configuration evaluations. Basically, how many instance-seed keys should be max evaluated
            for a configuration.
        max_incumbents : int, defaults to 10
            How many incumbents to keep track of in the case of multi-objective.
        """

        class NewIntensifier(
            intermediate_decision.NewCostDominatesOldCost, intermediate_update.ClosestIncumbentComparison, MOIntensifier
        ):
            pass

        return NewIntensifier(
            scenario=scenario,
            max_config_calls=max_config_calls,
            max_incumbents=max_incumbents,
        )

    @staticmethod
    # TODO update acquisition function with EIHV and PIHV
    def get_acquisition_function(  # type: ignore
        scenario: Scenario,
        *,
        xi: float = 0.0,
    ) -> EHVI:
        """Returns an Expected Improvement acquisition function.

        Parameters
        ----------
        scenario : Scenario
        xi : float, defaults to 0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        return PHVI()

    @staticmethod
    def get_acquisition_maximizer(  # type: ignore
        scenario: Scenario,
    ) -> MOLocalAndSortedRandomSearch:
        """Returns local and sorted random search as acquisition maximizer."""
        optimizer = MOLocalAndSortedRandomSearch(
            scenario.configspace,
            seed=scenario.seed,
        )

        return optimizer

    @staticmethod
    # TODO update initial design to LHD
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
    ) -> NoAggregationStrategy:
        """Returns the mean aggregation strategy for the multi objective algorithm.

        Parameters
        ----------
        scenario : Scenario
        objective_weights : list[float] | None, defaults to None
            Weights for averaging the objectives in a weighted manner. Must be of the same length as the number of
            objectives.
        """
        return NoAggregationStrategy()

    @staticmethod
    def get_runhistory_encoder(scenario: Scenario) -> RunHistoryEncoder:
        """Returns the default runhistory encoder with native multi objective support enabled."""
        return RunHistoryEncoder(scenario, native_multi_objective=True, normalize=False)
        # return RunHistoryLogEncoder(scenario, native_multi_objective=True, normalize=False)
