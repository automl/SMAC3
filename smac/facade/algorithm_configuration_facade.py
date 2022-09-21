from __future__ import annotations

from smac.acquisition.functions.expected_improvement import EI
from smac.acquisition.optimizers.local_and_random_search import LocalAndSortedRandomSearch
from smac.random_design.probability_design import ProbabilityRandomDesign
from ConfigSpace import Configuration
from smac.facade.abstract_facade import AbstractFacade
from smac.initial_design.default_design import DefaultInitialDesign
from smac.intensification.intensification import Intensifier
from smac.model.random_forest.random_forest import RandomForest
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
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
        bootstrapping: bool = True,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 20,
        pca_components: int = 4,
    ) -> RandomForest:
        """Returns a RandomForestWithInstances surrogate model. Please check its documentation."""
        return RandomForest(
            log_y=False,
            n_trees=n_trees,
            bootstrapping=bootstrapping,
            ratio_features=ratio_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            configspace=scenario.configspace,
            instance_features=scenario.instance_features,
            pca_components=pca_components,
            seed=scenario.seed,
        )

    @staticmethod
    def get_acquisition_function(  # type: ignore
        scenario: Scenario,
        xi: float = 0.0,
    ) -> EI:
        """Returns an Expected Improvement acquisition function. Please check its documentation."""
        return EI(xi=xi)

    @staticmethod
    def get_acquisition_optimizer(  # type: ignore
        scenario: Scenario,
    ) -> LocalAndSortedRandomSearch:
        optimizer = LocalAndSortedRandomSearch(
            scenario.configspace,
            seed=scenario.seed,
        )

        return optimizer

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        min_challenger=1,
        min_config_calls=1,
        max_config_calls=2000,
        intensify_percentage: float = 0.5,
    ) -> Intensifier:
        """Returns an Intensifier. Please check its documentation."""
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
        configs: list[Configuration] | None = None,
    ) -> DefaultInitialDesign:
        """Returns an DefaultInitialDesign, evaluating only the default configuration. Please check
        its documentation."""
        return DefaultInitialDesign(
            scenario=scenario,
            configs=configs,
        )

    @staticmethod
    def get_random_design(  # type: ignore
        scenario: Scenario,
        *,
        random_probability: float = 0.5,
    ) -> ProbabilityRandomDesign:
        """Returns a ProbabilityRandomDesign for interleaving configurations selected from BO
        with those of a RandomDesign. Please check its documentation."""
        return ProbabilityRandomDesign(probability=random_probability, seed=scenario.seed)

    @staticmethod
    def get_multi_objective_algorithm(  # type: ignore
        scenario: Scenario,
    ) -> MeanAggregationStrategy:
        """Returns a MultiObjectiveAlgorithm (MeanAggregationStrategy). Please check its
        documentation."""
        return MeanAggregationStrategy(scenario=scenario)

    @staticmethod
    def get_runhistory_encoder(scenario: Scenario) -> RunHistoryEncoder:
        return RunHistoryEncoder(scenario)
