from __future__ import annotations

from smac.acquisition_function import AbstractAcquisitionFunction
from smac.acquisition_function.expected_improvement import EI
from smac.acquisition_optimizer import AbstractAcquisitionOptimizer
from smac.acquisition_optimizer.local_and_random_search import (
    LocalAndSortedRandomSearch,
)
from smac.chooser.random_chooser import ChooserProb, RandomChooser
from smac.config import Config
from smac.configspace import Configuration
from smac.facade import Facade
from smac.initial_design import InitialDesign
from smac.initial_design.default_configuration_design import DefaultInitialDesign
from smac.intensification.intensification import Intensifier
from smac.model.random_forest.rf_with_instances import RandomForestWithInstances
from smac.model.utils import get_types
from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.runhistory.runhistory_transformer import RunhistoryTransformer
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class AlgorithmConfigurationFacade(Facade):
    @staticmethod
    def get_model(
        config: Config,
        *,
        n_trees: int = 10,
        bootstrapping: bool = True,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 20,
        pca_components: int = 4,
    ) -> RandomForestWithInstances:
        types, bounds = get_types(config.configspace, config.instance_features)

        return RandomForestWithInstances(
            types=types,
            bounds=bounds,
            log_y=False,
            num_trees=n_trees,
            do_bootstrapping=bootstrapping,
            ratio_features=ratio_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            configspace=config.configspace,
            instance_features=config.instance_features,
            pca_components=pca_components,
            seed=config.seed,
        )

    @staticmethod
    def get_acquisition_function(config: Config, par: float = 0.0) -> EI:
        return EI(par=par)

    @staticmethod
    def get_acquisition_optimizer(config: Config) -> AbstractAcquisitionOptimizer:
        optimizer = LocalAndSortedRandomSearch(
            config.configspace,
            seed=config.seed,
        )

        return optimizer

    @staticmethod
    def get_intensifier(
        config: Config,
        *,
        min_challenger=1,
        min_config_calls=1,
        max_config_calls=2000,
    ) -> Intensifier:
        if config.deterministic:
            min_challenger = 1

        intensifier = Intensifier(
            instances=config.instances,
            instance_specifics=config.instance_specifics,  # What is that?
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
    def get_initial_design(config: Config, *, initial_configs: list[Configuration] | None = None) -> InitialDesign:
        return DefaultInitialDesign(
            configspace=config.configspace,
            n_runs=config.n_runs,
            configs=initial_configs,
            n_configs_per_hyperparameter=0,
            seed=config.seed,
        )

    @staticmethod
    def get_random_configuration_chooser(config: Config, *, random_probability: float = 0.5) -> RandomChooser:
        return ChooserProb(prob=random_probability, seed=config.seed)

    @staticmethod
    def get_multi_objective_algorithm(config: Config) -> AbstractMultiObjectiveAlgorithm | None:
        if len(config.objectives) <= 1:
            return None

        return MeanAggregationStrategy(config.seed)

    @staticmethod
    def get_runhistory_transformer(config: Config):
        transformer = RunhistoryTransformer(
            config=config,
            n_params=len(config.configspace.get_hyperparameters()),
            scale_percentage=5,
            seed=config.seed,
        )

        return transformer
