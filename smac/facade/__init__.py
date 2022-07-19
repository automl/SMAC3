from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

import joblib
import numpy as np

from smac.acquisition_function import AbstractAcquisitionFunction
from smac.acquisition_optimizer import AbstractAcquisitionOptimizer
from smac.chooser.random_chooser import RandomChooser
from smac.config import Config
from smac.configspace import Configuration
from smac.initial_design import InitialDesign
from smac.intensification.abstract_racer import AbstractRacer
from smac.model.base_imputor import BaseImputor
from smac.model.base_model import BaseModel
from smac.model.random_forest.rf_with_instances import RandomForestWithInstances
from smac.model.random_forest.rfr_imputator import RFRImputator
from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.smbo import SMBO
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory_transformer import RunhistoryTransformer
from smac.runner.target_algorithm_runner import TargetAlgorithmRunner
from smac.runner import Runner
from smac.runner.dask_runner import DaskParallelRunner
from smac.utils.logging import get_logger
from smac.utils.others import recursively_compare_dicts
from smac.utils.stats import Stats

logger = get_logger(__name__)


class Facade:
    def __init__(
        self,
        config: Config,
        target_algorithm: Runner | Callable,
        *,
        model: BaseModel | None = None,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        acquisition_optimizer: AbstractAcquisitionOptimizer | None = None,
        initial_design: InitialDesign | None = None,
        random_configuration_chooser: RandomChooser | None = None,
        intensifier: AbstractRacer | None = None,
        multi_objective_algorithm: AbstractMultiObjectiveAlgorithm | None = None,
    ):
        if model is None:
            model = self.get_model(config)

        if acquisition_function is None:
            acquisition_function = self.get_acquisition_function(config)

        if acquisition_optimizer is None:
            acquisition_optimizer = self.get_acquisition_optimizer(config)

        if initial_design is None:
            initial_design = self.get_initial_design(config)

        if random_configuration_chooser is None:
            random_configuration_chooser = self.get_random_configuration_chooser(config)

        if intensifier is None:
            intensifier = self.get_intensifier(config)

        if multi_objective_algorithm is None:
            multi_objective_algorithm = self.get_multi_objective_algorithm(config)

        # Initialize empty stats and runhistory object
        stats = Stats(config)
        runhistory = RunHistory()

        # Set the seed for configuration space
        config.configspace.seed(config.seed)

        # Prepare the algorithm executer
        runner: Runner
        if callable(target_algorithm):
            # We wrap our algorithm with the AlgorithmExecuter to use pynisher
            # and to catch exceptions
            runner = TargetAlgorithmRunner(target_algorithm, stats=stats)
        elif isinstance(target_algorithm, Runner):
            runner = target_algorithm
        else:
            # TODO: Integrate ExecuteTARunOld again
            raise NotImplementedError

        # In case of multiple jobs, we need to wrap the runner again using `DaskParallelRunner`
        if (n_workers := config.n_workers) > 1:
            available_workers = joblib.cpu_count()
            if n_workers > available_workers:
                logger.info(f"Workers are reduced to {n_workers}.")
                n_workers = available_workers

            # We use a dask runner for parallelization
            runner = DaskParallelRunner(
                runner,
                n_workers=n_workers,
                output_directory=str(config.output_directory),
            )

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
        self.runhistory = runhistory
        self.runhistory_transformer = self.get_runhistory_transformer(config)
        self.stats = stats
        self.seed = config.seed

        # Create optimizer using the previously defined objects
        self.optimizer = SMBO(
            config=self.config,
            stats=self.stats,
            runner=self.runner,
            initial_design=self.initial_design,
            runhistory=self.runhistory,
            runhistory_transformer=self.runhistory_transformer,
            intensifier=self.intensifier,
            model=self.model,
            acquisition_function=self.acquisition_function,
            acquisition_optimizer=self.acquisition_optimizer,
            random_configuration_chooser=self.random_configuration_chooser,
            seed=self.seed,
        )

        # Adding dependencies of the components
        self._update_dependencies()

        # We have to update our meta data (basically arguments of the components)
        self.config._set_meta(self.get_meta())

        # We have to validate if the object compositions are correct and actually make sense
        self._validate()

        # Here we actually check whether the run should be continued or not.
        # More precisely, we update our stats and runhistory object if all kwargs
        # and config/stats object are the same. For doing so, we create a specific hash.
        # SMBO recognizes that stats is not empty and hence does not run initial design anymore.
        # Since the runhistory is already updated, the model uses previous data directly.
        self._continue()

        # And now we save our config object.
        # Runhistory and stats are saved by `SMBO` as they change over time.
        self.config.save()

    def _update_dependencies(self) -> None:
        # We add some more dependencies.
        # This is the easiest way to incorporate dependencies, although it might be a bit hacky.
        self.intensifier.set_stats(self.stats)
        self.runhistory_transformer.set_multi_objective_algorithm(self.multi_objective_algorithm)
        self.runhistory_transformer.set_imputer(self._get_imputer())
        self.acquisition_function.set_model(self.model)
        self.acquisition_optimizer.set_acquisition_function(self.acquisition_function)

        # TODO: self.runhistory_transformer.set_success_states etc. for different intensifier?

    def _validate(self) -> None:
        # Make sure the same acquisition function is used
        assert self.acquisition_function == self.acquisition_optimizer.acquisition_function

        # We have to check that if we use transform_y it's done everywhere
        # For example, if we have LogEI, we also need to transform the data inside RunhistoryTransformer

    def _continue(self) -> None:
        """Update the runhistory and stats object if configs (inclusive meta data) are the same."""
        old_output_directory = self.config.output_directory
        old_runhistory_filename = self.config.output_directory / "runhistory.json"
        old_stats_filename = self.config.output_directory / "stats.json"

        if old_output_directory.exists() and old_runhistory_filename.exists() and old_stats_filename.exists():
            old_config = Config.load(old_output_directory)

            if self.config == old_config:
                logger.info("Continuing from previous run.")

                # We update the runhistory and stats in-place.
                # Stats use the output directory from the config directly.
                self.runhistory.load_json(str(old_runhistory_filename), cs=self.config.configspace)
                self.stats.load()
            else:
                diff = recursively_compare_dicts(self.config.__dict__, old_config.__dict__, level="config")
                logger.info(
                    f"Found old run in `{self.config.output_directory}` but it is not the same as the current one:\n"
                    f"{diff}"
                )

                feedback = input(
                    "\nPress one of the following numbers to continue or any other key to abort:\n"
                    "(1) Overwrite old run completely.\n"
                    "(2) Overwrite old run and re-use previous runhistory data. The configuration space "
                    "has to be the same for this option.\n"
                )

                if feedback == "1":
                    # We don't have to do anything here, since we work with a clean runhistory and stats object
                    pass
                elif feedback == "2":
                    # We overwrite runhistory and stats.
                    # However, we should ensure that we use the same configspace.
                    assert self.config.configspace == old_config.configspace

                    self.runhistory.load_json(str(old_runhistory_filename), cs=self.config.configspace)
                    self.stats.load()
                else:
                    raise RuntimeError("SMAC run was stopped by the user.")

    def _get_imputer(self) -> BaseImputor | None:
        assert self.model is not None
        assert self.config

        # TODO: Can anyone tell me what the `par_factor` does?
        par_factor = 10.0

        if isinstance(self.model, RandomForestWithInstances):
            if self.model.log_y:
                algorithm_walltime_limit = np.log(np.nanmin([np.inf, np.float_(self.config.algorithm_walltime_limit)]))
                threshold = algorithm_walltime_limit + np.log(par_factor)
            else:
                algorithm_walltime_limit = np.nanmin([np.inf, np.float_(self.config.algorithm_walltime_limit)])
                threshold = algorithm_walltime_limit * par_factor

            return RFRImputator(
                model=self.model,
                algorithm_walltime_limit=algorithm_walltime_limit,
                max_iter=2,
                threshold=threshold,
                change_threshold=0.01,
                seed=self.seed,
            )

        return None

    def get_meta(self) -> dict[str, dict[str, Any]]:
        """Generates a hash based on all kwargs of the facade. This is used for determine
        whether a run should be continued or not."""
        meta = {
            "target_algorithm": {"code": str(self.runner.target_algorithm.__code__.co_code)},
            "initial_design": self.initial_design.get_meta(),
            # TODO: Create `get_meta` methods
            # "intensifier": self.intensifier.get_meta(),
            # "model": self.model.get_meta(),
            # "acquisition_function": self.acquisition_function.get_meta(),
            # "random_configuration_chooser": self.random_configuration_chooser.get_meta(),
        }

        return meta

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
                logger.info(f"Final Incumbent: {incumbent.get_dictionary()}")
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
    def get_acquisition_optimizer(config: Config) -> AbstractAcquisitionOptimizer:
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
