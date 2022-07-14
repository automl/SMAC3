from __future__ import annotations
from abc import abstractmethod
from typing import Callable
import dask
import joblib

import numpy as np
from smac.runner.algorithm_executer import AlgorithmExecuter
from smac.configspace import Configuration
from smac.acquisition import AbstractAcquisitionFunction
from smac.acquisition.maximizer import AbstractAcquisitionOptimizer
from smac.runner.base import BaseRunner
from smac.config import Config
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.abstract_racer import AbstractRacer
from smac.model.base_model import BaseModel
from smac.model.base_imputor import BaseImputor
from smac.chooser.random_chooser import RandomChooser
from smac.model.random_forest.rf_with_instances import RandomForestWithInstances
from smac.model.random_forest.rfr_imputator import RFRImputator
from smac.multi_objective.abstract_multi_objective_algorithm import AbstractMultiObjectiveAlgorithm
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory_transformer import RunhistoryTransformer
from smac.utils.logging import get_logger
from smac.utils.stats import Stats


logger = get_logger(__name__)


class Facade:
    def __init__(
        self,
        config: Config,
        target_algorithm: BaseRunner | Callable,
        *,
        model: BaseModel | None = None,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        acquisition_optimizer: AbstractAcquisitionOptimizer | None = None,
        initial_design: InitialDesign | None = None,
        random_configuration_chooser: RandomChooser | None = None,
        intensifier: AbstractRacer | None = None,
        multi_objective_algorithm: AbstractMultiObjectiveAlgorithm | None = None,
        run_id: int | None = None,
    ):
        # TODO: How to integrate `restore_incumbent`?

        if model is None:
            model = self.get_model(config)

        if acquisition_function is None:
            acquisition_function = self.get_acquisition_function(config)

        if acquisition_optimizer is None:
            acquisition_optimizer = self.get_acquisition_optimizer(config, acquisition_function)

        if initial_design is None:
            initial_design = self.get_initial_design(config)

        if random_configuration_chooser is None:
            random_configuration_chooser = self.get_random_configuration_chooser(config)

        if intensifier is None:
            intensifier = self.get_intensifier(config)

        if multi_objective_algorithm is None:
            multi_objective_algorithm = self.get_multi_objective_algorithm(config)

        stats = Stats(config)

        # Set the seed for configuration space
        config.configspace.seed(config.seed)

        # Prepare algorithm executer
        if callable(target_algorithm):
            # We wrap our algorithm with the AlgorithmExecuter to use pynisher
            # and to catch exceptions
            runner = AlgorithmExecuter(target_algorithm, stats=stats)
        elif isinstance(target_algorithm, BaseRunner):
            runner = target_algorithm
        else:
            # TODO: Integrate ExecuteTARunOld again
            raise NotImplementedError

        # In case of multiple jobs, we need to pass the algorithm to the dask client
        logger.info("FIXME: Parallelization is not working yet.")
        if config.n_workers == -1:
            n_workers = joblib.cpu_count()
        elif config.n_workers > 1:
            pass
            # algorithm = DaskParallelRunner(  # type: ignore
            #    algorithm,
            #    n_workers=n_workers,
            #    output_directory=config.output_directory,
            #    dask_client=dask_client,
            # )

        # Set variables globally
        self.config = config
        self.configspace = config.configspace
        self.runner = runner
        self.model = model
        self.acquisition_function = acquisition_function
        self.acquisition_optimizer = acquisition_optimizer
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
        self.runhistory_transformer.set_imputer(self._get_imputer())
        # self.runhistory_transformer.set_success_states etc. for different intensifier?

        # Unfortunately, we can't use the update method
        # here as update might incorporate increasing steps or else
        self.acquisition_function.set_model(model)

        if run_id is None:
            run_id = 0

        # Create optimizer
        self.optimizer = SMBO(
            config=config,
            stats=stats,
            runner=runner,
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

    def _validate(self) -> None:
        # Make sure the same acquisition function is used
        assert self.acquisition_function == self.acquisition_optimizer.acquisition_function

        # We have to check that if we use transform_y it's done everywhere
        # For example, if we have LogEI, we also need to transform the data inside RunhistoryTransformer

    def _get_imputer(self) -> BaseImputor | None:
        assert self.model is not None

        logger.error("FIX ME: HOW TO TRANSFORM Y?")
        return None

        if isinstance(self.model, RandomForestWithInstances):
            if self.log_y:
                cutoff = np.log(np.nanmin([np.inf, np.float_(self.config.algorithm_walltime_limit)]))
                threshold = cutoff + np.log(self.config.par_factor)
            else:
                cutoff = np.nanmin([np.inf, np.float_(self.config.algorithm_walltime_limit)])
                threshold = cutoff * self.config.par_factor

            return RFRImputator(
                model=self.model,
                algorithm_walltime_limit=self.config.algorithm_walltime_limit,
                max_iter=2,
                threshold=threshold,
                change_threshold=0.01,
                seed=self.seed,
            )

        return None

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

    @staticmethod
    @abstractmethod
    def get_model(config: Config) -> BaseModel:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_acquisition_function(config: Config) -> AbstractAcquisitionFunction:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_acquisition_optimizer(
        config: Config, acquisition_function: AbstractAcquisitionFunction
    ) -> AbstractAcquisitionOptimizer:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_intensifier(config: Config) -> AbstractRacer:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_initial_design(config: Config) -> InitialDesign:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_random_configuration_chooser(config: Config) -> RandomChooser:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_multi_objective_algorithm(config: Config) -> AbstractMultiObjectiveAlgorithm | None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_runhistory_transformer(config: Config) -> RunhistoryTransformer:
        raise NotImplementedError
