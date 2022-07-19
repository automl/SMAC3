from __future__ import annotations

from smac.acquisition_function.expected_improvement import EI
from smac.acquisition_optimizer.local_and_random_search import (
    LocalAndSortedRandomSearch,
)
from smac.chooser.random_chooser import ChooserProb
from smac.config import Config
from smac.configspace import Configuration
from smac.facade import Facade
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.intensification.intensification import Intensifier
from smac.model.random_forest.rf_with_instances import RandomForestWithInstances
from smac.model.utils import get_types
from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.runhistory.runhistory_transformer import RunhistoryLogScaledTransformer

__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class HyperparameterOptimizationFacade(Facade):
    @staticmethod
    def get_model(
        config: Config,
        *,
        n_trees: int = 10,
        bootstrapping: bool = True,
        ratio_features: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 2**20,
    ) -> RandomForestWithInstances:
        types, bounds = get_types(config.configspace)

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
            configspace=config.configspace,
            seed=config.seed,
        )

    @staticmethod
    def get_acquisition_function(config: Config, *, par: float = 0.0) -> EI:
        return EI(par=par, log=True)

    @staticmethod
    def get_acquisition_optimizer(
        config: Config,
        *,
        local_search_iterations: int = 10,
        challengers: int = 10000,
    ) -> LocalAndSortedRandomSearch:
        optimizer = LocalAndSortedRandomSearch(
            config.configspace,
            local_search_iterations=local_search_iterations,
            challengers=challengers,
            seed=config.seed,
        )

        return optimizer

    @staticmethod
    def get_intensifier(config: Config, *, min_challenger=1, min_config_calls=1, max_config_calls=3) -> Intensifier:
        intensifier = Intensifier(
            instances=config.instances,
            instance_specifics=config.instance_specifics,
            algorithm_walltime_limit=config.algorithm_walltime_limit,
            deterministic=config.deterministic,
            min_challenger=min_challenger,
            race_against=config.configspace.get_default_configuration(),
            min_config_calls=min_config_calls,
            max_config_calls=max_config_calls,
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
    ) -> SobolInitialDesign:
        return SobolInitialDesign(
            configspace=config.configspace,
            n_runs=config.n_runs,
            configs=initial_configs,
            n_configs_per_hyperparameter=n_configs_per_hyperparamter,
            max_config_ratio=max_config_ratio,
            seed=config.seed,
        )

    @staticmethod
    def get_random_configuration_chooser(config: Config, *, probability: float = 0.2) -> ChooserProb:
        return ChooserProb(prob=probability)

    @staticmethod
    def get_multi_objective_algorithm(config: Config) -> AbstractMultiObjectiveAlgorithm | None:
        if len(config.objectives) <= 1:
            return None

        return MeanAggregationStrategy(config.seed)

    @staticmethod
    def get_runhistory_transformer(config: Config) -> RunhistoryLogScaledTransformer:
        return RunhistoryLogScaledTransformer(
            config,
            n_params=len(config.configspace.get_hyperparameters()),
            seed=config.seed,
        )
