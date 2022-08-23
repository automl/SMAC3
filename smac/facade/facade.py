from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from pathlib import Path

import joblib
import numpy as np

from smac.acquisition.abstract_acqusition_optimizer import AbstractAcquisitionOptimizer
from smac.acquisition.functions.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.callback import Callback
from smac.random_design.random_design import RandomDesign
from smac.configspace import Configuration
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.model.base_model import BaseModel

# from smac.model.imputer import AbstractImputer
# from smac.model.imputer.random_forest_imputer import RandomForestImputer
from smac.model.random_forest.random_forest_with_instances import (
    RandomForestWithInstances,
)
from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.runhistory.runhistory import RunHistory
from smac.runner.dask_runner import DaskParallelRunner
from smac.runner.runner import Runner
from smac.runner.target_algorithm_runner import TargetAlgorithmRunner
from smac.scenario import Scenario
from smac.loop import SMBO
from smac.utils.data_structures import recursively_compare_dicts
from smac.utils.logging import get_logger, setup_logging
from smac.utils.stats import Stats

logger = get_logger(__name__)


class Facade:
    def __init__(
        self,
        scenario: Scenario,
        target_algorithm: Runner | Callable,
        *,
        model: BaseModel | None = None,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        acquisition_optimizer: AbstractAcquisitionOptimizer | None = None,
        initial_design: InitialDesign | None = None,
        random_design: RandomDesign | None = None,
        intensifier: AbstractIntensifier | None = None,
        multi_objective_algorithm: AbstractMultiObjectiveAlgorithm | None = None,
        runhistory: RunHistory | None = None,
        # Level of logging; if path passed: yaml file expected; if none: use default logging from logging.yml
        logging_level: int | Path | None = None,
        callbacks: list[Callback] = [],
        # Overwrites the results if they are already given; otherwise, the user is asked
        overwrite: bool = False,
    ):
        setup_logging(logging_level)

        if model is None:
            model = self.get_model(scenario)

        if acquisition_function is None:
            acquisition_function = self.get_acquisition_function(scenario)

        if acquisition_optimizer is None:
            acquisition_optimizer = self.get_acquisition_optimizer(scenario)

        if initial_design is None:
            initial_design = self.get_initial_design(scenario)

        if random_design is None:
            random_design = self.get_random_design(scenario)

        if intensifier is None:
            intensifier = self.get_intensifier(scenario)

        if multi_objective_algorithm is None and scenario.count_objectives() > 1:
            multi_objective_algorithm = self.get_multi_objective_algorithm(scenario)

        if runhistory is None:
            runhistory = RunHistory()

        # Initialize empty stats and runhistory object
        stats = Stats(scenario)

        # Set the seed for configuration space
        scenario.configspace.seed(scenario.seed)

        # Prepare the algorithm executer
        runner: Runner
        if callable(target_algorithm):
            # We wrap our algorithm with the AlgorithmExecuter to (potentially) use pynisher and catch exceptions
            runner = TargetAlgorithmRunner(
                target_algorithm,
                scenario=scenario,
                stats=stats,
            )
        elif isinstance(target_algorithm, Runner):
            runner = target_algorithm
        else:
            # TODO: Integrate ExecuteTARunOld again
            raise NotImplementedError

        # In case of multiple jobs, we need to wrap the runner again using `DaskParallelRunner`
        if (n_workers := scenario.n_workers) > 1:
            available_workers = joblib.cpu_count()
            if n_workers > available_workers:
                logger.info(f"Workers are reduced to {n_workers}.")
                n_workers = available_workers

            # We use a dask runner for parallelization
            runner = DaskParallelRunner(
                runner,
                n_workers=n_workers,
                output_directory=str(scenario.output_directory),
            )

        # Set variables globally
        self.scenario = scenario
        self.configspace = scenario.configspace
        self.runner = runner
        self.model = model
        self.acquisition_function = acquisition_function
        self.acquisition_optimizer = acquisition_optimizer
        self.initial_design = initial_design
        self.random_design = random_design
        self.intensifier = intensifier
        self.multi_objective_algorithm = multi_objective_algorithm
        self.runhistory = runhistory
        self.runhistory_encoder = self.get_runhistory_encoder(scenario)
        self.stats = stats
        self.seed = scenario.seed

        # Create optimizer using the previously defined objects
        self._init_optimizer()

        # Register callbacks here
        for callback in callbacks:
            self.optimizer.register_callback(callback)

        # Adding dependencies of the components
        self._update_dependencies()

        # We have to update our meta data (basically arguments of the components)
        self.scenario._set_meta(self.get_meta())

        # We have to validate if the object compositions are correct and actually make sense
        self._validate()

        # Here we actually check whether the run should be continued or not.
        # More precisely, we update our stats and runhistory object if all component arguments
        # and scenario/stats object are the same. For doing so, we create a specific hash.
        # The SMBO object recognizes that stats is not empty and hence does not the run initial design anymore.
        # Since the runhistory is already updated, the model uses previous data directly.
        self._continue(overwrite)

        # And now we save our scenario object.
        # Runhistory and stats are saved by `SMBO` as they change over time.
        self.scenario.save()

    def _init_optimizer(self) -> None:
        self.optimizer = SMBO(
            scenario=self.scenario,
            stats=self.stats,
            runner=self.runner,
            initial_design=self.initial_design,
            runhistory=self.runhistory,
            runhistory_encoder=self.runhistory_encoder,
            intensifier=self.intensifier,
            model=self.model,
            acquisition_function=self.acquisition_function,
            acquisition_optimizer=self.acquisition_optimizer,
            random_design=self.random_design,
            seed=self.seed,
        )

    def _update_dependencies(self) -> None:
        # We add some more dependencies.
        # This is the easiest way to incorporate dependencies, although it might be a bit hacky.
        self.intensifier._set_stats(self.stats)
        self.runhistory_encoder._set_multi_objective_algorithm(self.multi_objective_algorithm)
        # self.runhistory_encoder._set_imputer(self._get_imputer())
        self.acquisition_function._set_model(self.model)
        self.acquisition_optimizer._set_acquisition_function(self.acquisition_function)

        # TODO: self.runhistory_encoder.set_success_states etc. for different intensifier?

    def _validate(self) -> None:
        assert self.optimizer

        # Make sure the same acquisition function is used
        assert self.acquisition_function == self.acquisition_optimizer.acquisition_function

        # We have to check that if we use transform_y it's done everywhere
        # For example, if we have LogEI, we also need to transform the data inside RunHistoryEncoder

    def _continue(self, overwrite: bool = False) -> None:
        """Update the runhistory and stats object if configs (inclusive meta data) are the same."""
        if overwrite:
            return

        old_output_directory = self.scenario.output_directory
        old_runhistory_filename = self.scenario.output_directory / "runhistory.json"
        old_stats_filename = self.scenario.output_directory / "stats.json"

        if old_output_directory.exists() and old_runhistory_filename.exists() and old_stats_filename.exists():
            old_scenario = Scenario.load(old_output_directory)

            if self.scenario == old_scenario:
                logger.info("Continuing from previous run.")

                # We update the runhistory and stats in-place.
                # Stats use the output directory from the config directly.
                self.runhistory.reset()
                self.runhistory.load_json(str(old_runhistory_filename), configspace=self.scenario.configspace)
                self.stats.load()

                # Reset runhistory and stats if first run was not successful
                if self.stats.submitted == 1 and self.stats.finished == 0:
                    logger.info("Since the previous run was not successful, SMAC will start from scratch again.")
                    self.runhistory.reset()
                    self.stats.reset()
            else:
                diff = recursively_compare_dicts(self.scenario.__dict__, old_scenario.__dict__, level="scenario")
                logger.info(
                    f"Found old run in `{self.scenario.output_directory}` but it is not the same as the current one:\n"
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
                    assert self.scenario.configspace == old_scenario.configspace

                    self.runhistory.load_json(str(old_runhistory_filename), configspace=self.scenario.configspace)
                    self.stats.load()
                else:
                    raise RuntimeError("SMAC run was stopped by the user.")

    """
    def _get_imputer(self) -> AbstractImputer | None:
        assert self.model is not None
        assert self.scenario

        # TODO: Can anyone tell me what the `par_factor` does?
        par_factor = 10.0

        if isinstance(self.model, RandomForestWithInstances):
            if self.model.log_y:
                algorithm_walltime_limit = np.log(np.nanmin([np.inf, np.float_(self.scenario.trial_walltime_limit)]))
                threshold = algorithm_walltime_limit + np.log(par_factor)
            else:
                algorithm_walltime_limit = np.nanmin([np.inf, np.float_(self.scenario.trial_walltime_limit)])
                threshold = algorithm_walltime_limit * par_factor

            return RandomForestImputer(
                model=self.model,
                algorithm_walltime_limit=algorithm_walltime_limit,
                max_iter=2,
                threshold=threshold,
                change_threshold=0.01,
                seed=self.seed,
            )

        return None
    """

    def get_meta(self) -> dict[str, dict[str, Any]]:
        """Generates a hash based on all components of the facade. This is used for the run name or to determine
        whether a run should be continued or not."""

        multi_objective_algorithm_meta = None
        if self.multi_objective_algorithm is not None:
            multi_objective_algorithm_meta = self.multi_objective_algorithm.get_meta()

        meta = {
            "facade": {"name": self.__class__.__name__},
            "runner": self.runner.get_meta(),
            "model": self.model.get_meta(),
            "acquisition_optimizer": self.acquisition_optimizer.get_meta(),
            "acquisition_function": self.acquisition_function.get_meta(),
            "intensifier": self.intensifier.get_meta(),
            "initial_design": self.initial_design.get_meta(),
            "random_design": self.random_design.get_meta(),
            "runhistory_encoder": self.runhistory_encoder.get_meta(),
            "multi_objective_algorithm": multi_objective_algorithm_meta,
        }

        return meta

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
                logger.info(f"Final Incumbent: {incumbent.get_dictionary()}")
                logger.info(f"Estimated cost: {cost}")

        return incumbent

    @staticmethod
    @abstractmethod
    def get_model(scenario: Scenario) -> BaseModel:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_acquisition_function(scenario: Scenario) -> AbstractAcquisitionFunction:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_acquisition_optimizer(scenario: Scenario) -> AbstractAcquisitionOptimizer:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_intensifier(scenario: Scenario) -> AbstractIntensifier:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_initial_design(scenario: Scenario) -> InitialDesign:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_random_design(scenario: Scenario) -> RandomDesign:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_runhistory_encoder(scenario: Scenario) -> RunHistoryEncoder:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_multi_objective_algorithm(scenario: Scenario) -> AbstractMultiObjectiveAlgorithm:
        raise NotImplementedError
