from __future__ import annotations

from typing import Callable

import dask.distributed  # type: ignore
import joblib  # type: ignore

from smac.algorithm.algorithm_executer import AlgorithmExecuter

# tae
from smac.algorithm.base import BaseRunner
from smac.algorithm.dask_runner import DaskParallelRunner
from smac.config import Config
from smac.configspace import Configuration
from smac.epm.base_epm import BaseEPM

# epm
from smac.epm.random_forest.rf_with_instances import RandomForestWithInstances
from smac.epm.utils import get_types
from smac.initial_design.default_configuration_design import DefaultInitialDesign

# Initial designs
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.abstract_racer import AbstractRacer

# intensification
from smac.intensification.intensification import Intensifier
from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.optimizer.acquisition import EI, AbstractAcquisitionFunction
from smac.optimizer.acquisition.maximizer import (
    AbstractAcquisitionOptimizer,
    LocalAndSortedRandomSearch,
)
from smac.optimizer.configuration_chooser.random_chooser import (
    ChooserProb,
    RandomChooser,
)

# optimizer
from smac.optimizer.smbo import SMBO

# runhistory
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory_transformer import RunhistoryTransformer

# utils
from smac.utils.logging import get_logger

# stats and options
from smac.utils.stats import Stats

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


# SMAC4AC -> AlgorithmConfigurationFacade
# SMAC4BB -> BlackBoxFacade
# SMAC4MF -> MultiFidelityFacade
# ROAR -> RandomFacade
class AlgorithmConfigurationFacade:
    def __init__(
        self,
        config: Config,
        algorithm: BaseRunner | Callable,
        *,
        model: BaseEPM | None = None,  # Optimizer model
        acquisition_function: AbstractAcquisitionFunction | None = None,
        acquisition_optimizer: AbstractAcquisitionOptimizer | None = None,
        initial_design: InitialDesign | None = None,
        random_configuration_chooser: RandomChooser | None = None,
        intensifier: AbstractRacer | None = None,
        multi_objective_algorithm: AbstractMultiObjectiveAlgorithm | None = None,
        run_id: int | None = None,
        dask_client: dask.distributed.Client | None = None,
        n_workers: int = 1,
        stats: Stats | None = None,
    ):
        # TODO: How to integrate `restore_incumbent`?
        if model is None:
            model = self.get_model(config)

        if acquisition_function is None:
            acquisition_function = self.get_acquisition_function(config)

        if acquisition_optimizer is None:
            acquisition_function_optimizer = self.get_acquisition_optimizer(config, acquisition_function)

        if initial_design is None:
            initial_design = self.get_initial_design(config)

        if random_configuration_chooser is None:
            random_configuration_chooser = self.get_random_configuration_chooser(config)

        if intensifier is None:
            intensifier = self.get_intensifier(config)

        if multi_objective_algorithm is None:
            multi_objective_algorithm = self.get_multi_objective_algorithm(config)

        if stats is None:
            stats = Stats(config)

        # Set the seed for configuration space
        config.configspace.seed(config.seed)
        exit()

        # Prepare algorithm executer
        if callable(algorithm):
            # We wrap our algorithm with the AlgorithmExecuter to use pynisher
            # and to catch exceptions
            algorithm = AlgorithmExecuter(algorithm, config=config, stats=stats)
        else:
            # TODO: Integrate ExecuteTARunOld again
            raise NotImplementedError

        # In case of multiple jobs, we need to pass the algorithm to the dask client
        if n_workers == -1:
            n_workers = joblib.cpu_count()
        elif n_workers > 1:
            algorithm = DaskParallelRunner(  # type: ignore
                algorithm,
                n_workers=n_workers,
                output_directory=config.output_directory,
                dask_client=dask_client,
            )

        # Set variables globally
        self.config = config
        self.configspace = config.configspace
        self.algorithm = algorithm
        self.model = model
        self.acquisition_function = acquisition_function
        self.acquisition_function_optimizer = acquisition_function_optimizer
        self.initial_design = initial_design
        self.random_configuration_chooser = random_configuration_chooser
        self.intensifier = intensifier
        self.multi_objective_algorithm = multi_objective_algorithm
        self.runhistory = RunHistory()
        self.runhistory_transformer = self.get_runhistory_transformer(config)
        self.stats = stats
        self.seed = config.seed

        # We have to validate if the object compositions are correct and actually make sense
        self._validate()

        # Now we add some more dependencies
        # This is the easiest way to do this, although it might be a bit hacky
        self.intensifier.set_stats(stats)
        self.runhistory_transformer.set_multi_objective_algorithm(self.multi_objective_algorithm)
        self.runhistory_transformer.set_imputer(self.model.get_imptuter())
        # self.runhistory_transformer.set_success_states etc. for different intensifier?

        # Unfortunately, we can't use the update method
        # here as update might incorporate increasing steps or else
        self.acquisition_function.set_model(model)

        # Create optimizer
        self.optimizer = SMBO(
            config=config,
            stats=stats,
            algorithm=algorithm,
            initial_design=initial_design,
            runhistory=self.runhistory,
            runhistory_transformer=self.runhistory_transformer,
            intensifier=intensifier,
            run_id=run_id,
            model=model,
            acquisition_function=acquisition_function,
            acquisition_optimizer=acquisition_optimizer,
            random_configuration_chooser=random_configuration_chooser,
            seed=self.seed,
        )

    def _validate(self):
        # Make sure the same acquisition function is used
        assert self.acquisition_function == self.acquisition_function_optimizer.acquisition_function

        # We have to check that if we use transform_y it's done everywhere
        # For example, if we have LogEI, we also need to transform the data inside RunhistoryTransformer

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
    def get_acquisition_function(config: Config) -> AbstractAcquisitionFunction:
        return EI()

    @staticmethod
    def get_acquisition_optimizer(
        config: Config, acquisition_function: AbstractAcquisitionFunction
    ) -> AbstractAcquisitionOptimizer:
        optimizer = LocalAndSortedRandomSearch(
            acquisition_function,
            config.configspace,
            seed=config.seed,
        )

        return optimizer

    @staticmethod
    def get_intensifier(
        config: Config,
        *,
        intensification_percentage: float = 0.5,
        adaptive_capping_slackfactor: float = 1.2,
        min_challenger=1,
        min_config_calls=1,
        max_config_calls=2000,
    ) -> Intensifier:
        if config.determinstic:
            min_challenger = 1

        intensifier = Intensifier(
            train_instances=config.train_instances,
            instance_specifics=config.instance_specifics,  # What is that?
            algorithm_walltime_limit=config.algorithm_walltime_limit,
            deterministic=config.deterministic,
            adaptive_capping_slackfactor=adaptive_capping_slackfactor,
            min_challenger=min_challenger,
            race_against=config.configspace.get_default_configuration(),
            min_config_calls=min_config_calls,
            max_config_calls=max_config_calls,
            seed=config.seed,
        )

        return intensifier

    @staticmethod
    def get_initial_design(config: Config, *, initial_configs: List[Configuration] | None = None) -> InitialDesign:
        return DefaultInitialDesign(
            configspace=config.configspace,
            n_runs=config.n_runs,
            configs=initial_configs,
            n_configs_per_hyperparameter=0,
            seed=config.seed,
        )

    @staticmethod
    def get_random_configuration_chooser(config: Config, *, random_probability: float = 0.5) -> RandomChooser:
        return ChooserProb(config.seed, random_probability)

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

    def optimize(self, save_instantly: bool = True) -> Configuration:
        """
        Optimizes the algorithm.

        Returns
        -------
        incumbent : Configuration
            Best found configuration.
        """
        incumbent = None
        try:
            incumbent = self.optimizer.run()
        finally:
            self.optimizer.save()
            self.stats.print()

            if incumbent is not None:
                cost = self.runhistory.get_cost(incumbent)
                logger.info(f"Final Incumbent: {incumbent}")
                logger.info(f"Estimated cost: {cost}")

        return incumbent

    def register_callback(self, callback: Callable) -> None:
        """Register a callback function."""
        raise NotImplementedError
