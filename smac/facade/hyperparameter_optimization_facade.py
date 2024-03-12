from __future__ import annotations

from ConfigSpace import Configuration

from smac.acquisition.function.expected_improvement import EI
from smac.acquisition.maximizer.local_and_random_search import (
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


class HyperparameterOptimizationFacade(AbstractFacade):
    @staticmethod
    def get_model(  # type: ignore
        scenario: Scenario,
        *,
        n_trees: int = 10,
        ratio_features: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 2**20,
        bootstrapping: bool = True,
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
        """
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
        """Returns an Expected Improvement acquisition function.

        Parameters
        ----------
        scenario : Scenario
        xi : float, defaults to 0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        return EI(xi=xi, log=True)

    @staticmethod
    def get_acquisition_maximizer(  # type: ignore
        scenario: Scenario,
        *,
        challengers: int = 10000,
        local_search_iterations: int = 10,
    ) -> LocalAndSortedRandomSearch:
        """Returns local and sorted random search as acquisition maximizer.

        Warning
        -------
        If you experience RAM issues, try to reduce the number of challengers.

        Parameters
        ----------
        challengers : int, defaults to 10000
            Number of challengers.
        local_search_iterations: int, defaults to 10
            Number of local search iterations.
        """
        optimizer = LocalAndSortedRandomSearch(
            scenario.configspace,
            challengers=challengers,
            local_search_iterations=local_search_iterations,
            seed=scenario.seed,
        )

        return optimizer

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        max_config_calls: int = 3,
        max_incumbents: int = 10,
    ) -> Intensifier:
        """Returns ``Intensifier`` as intensifier. Uses the default configuration for ``race_against``.

        Parameters
        ----------
        scenario : Scenario
        max_config_calls : int, defaults to 3
            Maximum number of configuration evaluations. Basically, how many instance-seed keys should be max evaluated
            for a configuration.
        max_incumbents : int, defaults to 10
            How many incumbents to keep track of in the case of multi-objective.
        """
        return Intensifier(
            scenario=scenario,
            max_config_calls=max_config_calls,
            max_incumbents=max_incumbents,
        )

    @staticmethod
    def get_initial_design(  # type: ignore
        scenario: Scenario,
        *,
        n_configs: int | None = None,
        n_configs_per_hyperparamter: int = 10,
        max_ratio: float = 0.25,
        additional_configs: list[Configuration] | None = None,
    ) -> SobolInitialDesign:
        """Returns a Sobol design instance.

        Parameters
        ----------
        scenario : Scenario
        n_configs : int | None, defaults to None
            Number of initial configurations (disables the arguments ``n_configs_per_hyperparameter``).
        n_configs_per_hyperparameter: int, defaults to 10
            Number of initial configurations per hyperparameter. For example, if my configuration space covers five
            hyperparameters and ``n_configs_per_hyperparameter`` is set to 10, then 50 initial configurations will be
            samples.
        max_ratio: float, defaults to 0.25
            Use at most ``scenario.n_trials`` * ``max_ratio`` number of configurations in the initial design.
            Additional configurations are not affected by this parameter.
        additional_configs: list[Configuration], defaults to []
            Adds additional configurations to the initial design.
        """
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
        """Returns ``ProbabilityRandomDesign`` for interleaving configurations.

        Parameters
        ----------
        probability : float, defaults to 0.2
            Probability that a configuration will be drawn at random.
        """
        return ProbabilityRandomDesign(probability=probability, seed=scenario.seed)

    @staticmethod
    def get_multi_objective_algorithm(  # type: ignore
        scenario: Scenario,
        *,
        objective_weights: list[float] | None = None,
    ) -> MeanAggregationStrategy:
        """Returns the mean aggregation strategy for the multi-objective algorithm.

        Parameters
        ----------
        scenario : Scenario
        objective_weights : list[float] | None, defaults to None
            Weights for averaging the objectives in a weighted manner. Must be of the same length as the number of
            objectives.
        """
        return MeanAggregationStrategy(
            scenario=scenario,
            objective_weights=objective_weights,
        )

    @staticmethod
    def get_runhistory_encoder(  # type: ignore
        scenario: Scenario,
    ) -> RunHistoryLogScaledEncoder:
        """Returns a log scaled runhistory encoder. That means that costs are log scaled before
        training the surrogate model.
        """
        return RunHistoryLogScaledEncoder(scenario)
