from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

import inspect
import logging

import dask.distributed  # type: ignore
import joblib  # type: ignore
import numpy as np

from smac.cli.output_directory import create_output_directory
from smac.cli.scenario import Scenario
from smac.cli.traj_logging import TrajEntry, TrajLogger
from smac.config import Config
from smac.configspace import Configuration
from smac.constants import MAXINT
from smac.epm.base_epm import BaseEPM
from smac.epm.multi_objective_epm import MultiObjectiveEPM

# epm
from smac.epm.random_forest.rf_with_instances import RandomForestWithInstances
from smac.epm.random_forest.rfr_imputator import RFRImputator
from smac.epm.utils import get_rng, get_types
from smac.initial_design.default_configuration_design import DefaultInitialDesign
from smac.initial_design.factorial_design import FactorialInitialDesign

# Initial designs
from smac.initial_design.initial_design import InitialDesign
from smac.initial_design.latin_hypercube_design import LatinHypercubeInitialDesign
from smac.initial_design.random_configuration_design import RandomInitialDesign
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.intensification.abstract_racer import AbstractRacer
from smac.intensification.hyperband import Hyperband

# intensification
from smac.intensification.intensification import Intensifier
from smac.intensification.successive_halving import SuccessiveHalving
from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.multi_objective.aggregation_strategy import (
    AggregationStrategy,
    MeanAggregationStrategy,
)
from smac.optimizer.acquisition import (
    EI,
    EIPS,
    AbstractAcquisitionFunction,
    IntegratedAcquisitionFunction,
    LogEI,
    PriorAcquisitionFunction,
)
from smac.optimizer.acquisition.maximizer import (
    AbstractAcquisitionFunctionOptimizer,
    LocalAndSortedPriorRandomSearch,
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
from smac.runhistory.runhistory2epm import (
    AbstractRunHistory2EPM,
    RunhistoryTransformer,
    RunhistoryInverseScaledTransformer,
    RunhistoryLogTransformer,
    RunhistoryLogScaledTransformer,
)

# stats and options
from smac.stats.stats import Stats
from smac.tae import StatusType

# tae
from smac.tae.base import BaseRunner
from smac.tae.dask_runner import DaskParallelRunner
from smac.tae.execute_func import AlgorithmExecuter
from smac.tae.execute_ta_run_old import ExecuteTARunOld

# utils
from smac.utils.logging import format_array, get_logger

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class AlgorithmConfiguration:
    def __init__(
        self,
        config: Config,
        algorithm: BaseRunner | Callable,
        *,
        model: BaseEPM | None = None,  # Optimizer model
        acquisition_function: AbstractAcquisitionFunction | None = None,
        acquisition_function_optimizer: AbstractAcquisitionFunctionOptimizer | None = None,
        initial_design: InitialDesign = None,
        random_configuration_chooser: RandomChooser | None = None,
        intensifier: AbstractRacer | None = None,
        multi_objective_algorithm: AbstractMultiObjectiveAlgorithm | None = None,
        run_id: Optional[int] = None,
        dask_client: dask.distributed.Client | None = None,
        n_workers: int = 1,
        stats: Stats | None = None,
    ):
        # TODO: How to integrate `restore_incumbent`?
        if model is None:
            model = self.get_model(config)

        if acquisition_function is None or acquisition_function_optimizer is None:
            if acquisition_function is not None:
                logger.info("Passed acquistion function is overwritten.")
            if acquisition_function_optimizer is not None:
                logger.info("Passed acquisition function optimizer is overwritten.")

            acquisition_function, acquisition_function_optimizer = self.get_acquisition(config, model)

        if initial_design is None:
            initial_design = self.get_initial_design(config)

        if random_configuration_chooser is None:
            random_configuration_chooser = self.get_random_configuration_chooser(config)

        if intensifier is None:
            intensifier = self.get_intensifier(config)

        if multi_objective_algorithm is None:
            multi_objective_algorithm = self.get_multi_objective_algorithm(config)

        # Set the seed for configuration space
        config.configspace.seed(config.seed)

        # Prepare algorithm executer
        if callable(algorithm):
            # We wrap our algorithm with the AlgorithmExecuter to use pynisher
            # and to catch exceptions
            algorithm = AlgorithmExecuter(algorithm, config=config, stats=self.stats)
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
        self.runhistory_transformer = self.get_runhistory_transformer(config, model)
        self.stats = Stats(config)
        self.seed = config.seed

        # Now we add some more dependencies
        self.intensifier.set_stats(self.stats)

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
            acquisition_function_optimizer=acquisition_function_optimizer,
            random_configuration_chooser=random_configuration_chooser,
            seed=self.seed,
        )

    @staticmethod
    def get_model(
        config: Config,
        *,
        n_trees: int,
        bootstrapping: bool,
        ratio_features: float,
        min_samples_split: float,
        min_samples_leaf: int,
        max_depth: int,
        pca_components: int = 4,
    ) -> RandomForestWithInstances:
        types, bounds = get_types(config.configspace, config.instance_features)
        log_y = "log" in config.transform_y

        return RandomForestWithInstances(
            types=types,
            bounds=bounds,
            log_y=log_y,
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
    def get_acquisition(
        config: Config, model: BaseEPM, *, integrated: bool = False, priors: bool = False
    ) -> tuple[AbstractAcquisitionFunction, AbstractAcquisitionFunctionOptimizer]:
        """Integrate max_steps_local_search/n_steps_plateau_walk/?"""
        if "log" in config.transform_y:
            acquisition_function: AbstractAcquisitionFunction = LogEI(model)
        else:
            acquisition_function = EI(model)

        if integrated:
            acquisition_function = IntegratedAcquisitionFunction(
                model,
                acquisition_function,
            )

        if priors:
            # A solid default value for decay_beta (empirically founded).
            default_beta = config.n_runs / 10
            discretize = isinstance(model, (RandomForestWithInstances, RFRImputator))

            acquisition_function = PriorAcquisitionFunction(
                model,
                acquisition_function,
                decay_beta=default_beta,
                discretize=discretize,
            )
            acquisition_function_optimizer = LocalAndSortedPriorRandomSearch(
                acquisition_function,
                configspace=config.configspace,
                uniform_configspace=config.configspace.remove_hyperparameter_priors(),
                seed=config.seed,
            )
        else:
            acquisition_function_optimizer = LocalAndSortedRandomSearch(
                acquisition_function,
                config.configspace,
                seed=config.seed,
            )

        return acquisition_function, acquisition_function_optimizer

    @staticmethod
    def get_intensifier(
        config: Config,
        method="default",
        *,
        intensification_percentage: float = 0.5,
        adaptive_capping_slackfactor: float = 1.2,
        min_challenger=1,
        min_config_calls=1,
        max_config_calls=2000,
    ) -> Intensifier:
        # TODO: Potentially add all intensifier here?
        # Nah, I guess going for one single intensifier is more than sufficient.
        # Other facade can define other intensifiers there.
        # And if the user needs even more flexibility, he can select another intensifier
        # directly.

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
    def get_initial_design(
        config: Config, method: str = "default", *, initial_configs: List[Configuration] | None = None
    ) -> InitialDesign:
        # TODO: Remove `method` and only use defaultinitialdesign? Each facade can
        # design its own initial design.
        # The idea is more like to give the user the possibility to change default parameters
        # for the chosen class for a facade.

        classes = {
            "default": DefaultInitialDesign,
            "random": RandomInitialDesign,
            "latin": LatinHypercubeInitialDesign,
            "factorial": FactorialInitialDesign,
            "sobol": SobolInitialDesign,
        }

        if method not in classes:
            raise ValueError(f"Invalid initialization method {method}.")

        kwargs = {
            "configspace": config.configspace,
            "n_runs": config.n_runs,
            "configs": initial_configs,
            "n_runs": config.n_runs,
            "n_configs_per_hyperparameter": 0,
            # "max_config_fracs": 0.0,
            "seed": config.seed,
        }

        if initial_configs is not None:
            initial_design = InitialDesign(**kwargs)
        else:
            initial_design = classes[method](**kwargs)

        return initial_design

    @staticmethod
    def get_random_configuration_chooser(config: Config, *, random_probability: float = 0.5) -> RandomChooser:
        return ChooserProb(config.seed, random_probability)

    @staticmethod
    def get_multi_objective_algorithm(config: Config) -> AbstractMultiObjectiveAlgorithm | None:
        if len(config.objectives) <= 1:
            return None

        return MeanAggregationStrategy(config.seed)

    @staticmethod
    def get_runhistory_transformer(config: Config, model: BaseEPM):
        imputer = None
        if inspect.isclass(model) == RandomForestWithInstances:
            # If we log the performance data, the RFRImputator will already get log transform data from the runhistory.
            if "log" in config.transform_y:
                cutoff = np.log(np.nanmin([np.inf, np.float_(config.algorithm_walltime_limit)]))
                threshold = cutoff + np.log(config.par_factor)
            else:
                cutoff = np.nanmin([np.inf, np.float_(config.algorithm_walltime_limit)])
                threshold = cutoff * config.par_factor

            imputer = RFRImputator(
                model=model,
                algorithm_walltime_limit=config.algorithm_walltime_limit,
                max_iter=2,
                threshold=threshold,
                change_threshold=0.01,
                seed=config.seed,
            )

        kwargs = {
            "config": config,
            "n_params": len(config.configspace.get_hyperparameters()),
            "success_states": [StatusType.SUCCESS, StatusType.CRASHED, StatusType.MEMOUT],
            "imputor": imputer,
            "impute_state": None,
            "impute_censored_data": False,
            "scale_perc": 5,
            # "multi_objective_algorithm": self.multi_objective_algorithm,
        }

        # TODO: Do we really need that?
        if isinstance(self.intensifier, (SuccessiveHalving, Hyperband)):
            kwargs.update(
                {
                    "success_states": [
                        StatusType.SUCCESS,
                        StatusType.CRASHED,
                        StatusType.MEMOUT,
                        StatusType.DONOTADVANCE,
                    ],
                    "consider_for_higher_budgets_state": [
                        StatusType.DONOTADVANCE,
                        StatusType.TIMEOUT,
                        StatusType.CRASHED,
                        StatusType.MEMOUT,
                    ],
                }
            )

        if config.transform_y is None:
            transformer = RunhistoryTransformer(**kwargs)
        elif config.transform_y == "log":
            transformer = RunhistoryLogTransformer(**kwargs)
        elif config.transform_y == "log_scaled":
            transformer = RunhistoryLogScaledTransformer(**kwargs)
        elif config.transform_y == "inverse_scaled":
            transformer = RunhistoryInverseScaledTransformer(**kwargs)

        return transformer

    def optimize(self) -> Configuration:
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
