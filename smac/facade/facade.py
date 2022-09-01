from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Iterator

from pathlib import Path

import joblib

import smac
from smac.acquisition.abstract_acqusition_optimizer import AbstractAcquisitionOptimizer
from smac.acquisition.functions.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.callback import Callback
from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.configspace import Configuration
from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.model.abstract_model import AbstractModel

from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runhistory.encoder.abstract_encoder import AbstractRunHistoryEncoder
from smac.runhistory.enumerations import TrialInfoIntent
from smac.runhistory.runhistory import RunHistory
from smac.runner.dask_runner import DaskParallelRunner
from smac.runner.runner import AbstractRunner
from smac.runner.target_algorithm_runner import TargetAlgorithmRunner
from smac.scenario import Scenario
from smac.main import SMBO
from smac.utils.logging import get_logger, setup_logging
from smac.stats import Stats

logger = get_logger(__name__)

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class Facade:
    """Facade is an abstraction on top of the SMBO backend to organize the components
    of a Bayesian Optimization loop in a configurable & separable manner to suit the
    various needs of different hyperparameter optimization pipelines.

    With the exception to scenario & target_algorithm, which are expected of the
    user, the parameters: model, aacquisition_function, acquisition_optimizer,
    initial_design, random_design, intensifier, multi_objective_algorithm,
    runhistory and runhistory_encoder can either be explicitly specified in the
    subclasses' get_* methods - defining a specific BO pipeline - or be instantiated
    by the user to overwrite a pipelines components explicitly, before passing
    them to the facade. For an example of the latter see the svm_cv.py.

    Parameters
    ----------
    scenario: Scenario,
    target_algorithm: AbstractRunner | Callable

    model: BaseModel | None
    acquisition_function: AbstractAcquisitionFunction | None
    acquisition_optimizer: AbstractAcquisitionOptimizer | None
    initial_design: InitialDesign | None
    random_design: RandomDesign | None
    intensifier: AbstractIntensifier | None
    multi_objective_algorithm: AbstractMultiObjectiveAlgorithm | None
    runhistory: RunHistory | None
    runhistory_encoder: RunHistoryEncoder | None

    logging_level: int | Path | None
         Level of logging; if path passed: yaml file expected; if none: use default logging from logging.yml
    callbacks: list[Callback] = [],
    overwrite: bool defaults to False
        When True, overwrites the results (Runhistory & Stats) if a previous run is found that is
        inconsistent in the meta data with the current setup. In that case, the user is prompted
        for the exact behaviour.
    """

    def __init__(
        self,
        scenario: Scenario,
        target_algorithm: AbstractRunner | Callable,
        *,
        model: AbstractModel | None = None,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        acquisition_optimizer: AbstractAcquisitionOptimizer | None = None,
        initial_design: AbstractInitialDesign | None = None,
        random_design: AbstractRandomDesign | None = None,
        intensifier: AbstractIntensifier | None = None,
        multi_objective_algorithm: AbstractMultiObjectiveAlgorithm | None = None,
        runhistory_encoder: AbstractRunHistoryEncoder | None = None,
        logging_level: int | Path | None = None,
        callbacks: list[Callback] = [],
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
            multi_objective_algorithm = self.get_multi_objective_algorithm(scenario=scenario)

        if runhistory_encoder is None:
            runhistory_encoder = self.get_runhistory_encoder(scenario)

        # Initialize empty stats and runhistory object
        runhistory = RunHistory()
        stats = Stats(scenario)

        # Set the seed for configuration space
        scenario.configspace.seed(scenario.seed)

        # Prepare the algorithm executer
        runner: AbstractRunner
        if callable(target_algorithm):
            # We wrap our algorithm with the AlgorithmExecuter to (potentially) use pynisher and catch exceptions
            runner = TargetAlgorithmRunner(
                target_algorithm,
                scenario=scenario,
                stats=stats,
            )
        elif isinstance(target_algorithm, AbstractRunner):
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
            runner = DaskParallelRunner(single_worker=runner)

        # Set variables globally
        self._scenario = scenario
        self._runner = runner
        self._model = model
        self._acquisition_function = acquisition_function
        self._acquisition_optimizer = acquisition_optimizer
        self._initial_design = initial_design
        self._random_design = random_design
        self._intensifier = intensifier
        self._multi_objective_algorithm = multi_objective_algorithm
        self._runhistory = runhistory
        self._runhistory_encoder = runhistory_encoder
        self._stats = stats
        self._overwrite = overwrite
        self._optimizer = self._get_optimizer()

        # Register callbacks here
        for callback in callbacks:
            self._optimizer._register_callback(callback)

        # Adding dependencies of the components
        self._update_dependencies()

        # We have to update our meta data (basically arguments of the components)
        self._scenario._set_meta(self.get_meta())

        # We have to validate if the object compositions are correct and actually make sense
        self._validate()

    @property
    def runhistory(self) -> RunHistory:
        return self._optimizer._runhistory

    @property
    def stats(self) -> Stats:
        return self._optimizer._stats

    def get_meta(self) -> dict[str, Any]:
        """Generates a hash based on all components of the facade. This is used for the run name or to determine
        whether a run should be continued or not."""

        multi_objective_algorithm_meta = None
        if self._multi_objective_algorithm is not None:
            multi_objective_algorithm_meta = self._multi_objective_algorithm.get_meta()

        meta = {
            "facade": {"name": self.__class__.__name__},
            "runner": self._runner.get_meta(),
            "model": self._model.get_meta(),
            "acquisition_optimizer": self._acquisition_optimizer.get_meta(),
            "acquisition_function": self._acquisition_function.get_meta(),
            "intensifier": self._intensifier.get_meta(),
            "initial_design": self._initial_design.get_meta(),
            "random_design": self._random_design.get_meta(),
            "runhistory_encoder": self._runhistory_encoder.get_meta(),
            "multi_objective_algorithm": multi_objective_algorithm_meta,
            "version": smac.version,
        }

        return meta

    def get_next_configurations(self) -> Iterator[Configuration]:
        """Choose next candidate solution with Bayesian optimization. The suggested configurations
        depend on the surrogate model acquisition optimizer/function. This method is used by
        the intensifier."""
        return self._optimizer.get_next_configurations()

    def ask(self) -> TrialInfo:
        """Asks the intensifier for the next trial. This method returns only trials with the intend
        to run."""
        while True:
            intend, info = self._optimizer.ask()
            # We only accept trials which are intented to run
            if intend != TrialInfoIntent.RUN:
                continue

            return info

    def tell(self, info: TrialInfo, value: TrialValue, time_left: float | None = None, save: bool = True) -> None:
        """Adds the result of a trial to the runhistory and updates the intensifier. Also,
        the stats object is updated.

        Parameters
        ----------
        info: TrialInfo
            Describes the trial from which to process the results.
        value: TrialValue
            Contains relevant information regarding the execution of a trial.
        time_left: float | None
            How much time in seconds is left to perform intensification.
        save : bool, optional to True
            Whether the runhistory should be saved.
        """
        return self._optimizer.tell(info, value, time_left, save)

    def optimize(self, force_initial_design: bool = False) -> Configuration:
        """
        Optimizes the algorithm.

        Parameters
        ----------
        force_initial_design: bool
            The initial design is only performed if the runhistory is empty. If this flag is set to True,
            the initial design is performed regardless of the runhistory.

        Returns
        -------
        incumbent : Configuration
            Best found configuration.
        """
        incumbent = None
        try:
            incumbent = self._optimizer.run(force_initial_design=force_initial_design)
        finally:
            self._optimizer.save()
            self._stats.print()

            if incumbent is not None:
                cost = self._runhistory.get_cost(incumbent)
                logger.info(f"Final Incumbent: {incumbent.get_dictionary()}")
                logger.info(f"Estimated cost: {cost}")

        return incumbent

    @staticmethod
    @abstractmethod
    def get_model(scenario: Scenario) -> AbstractModel:
        """Returns the surrogate cost model instance used in the BO loop."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_acquisition_function(scenario: Scenario) -> AbstractAcquisitionFunction:
        """Returns the acquisition function instance used in the BO loop,
        defining the exploration/exploitation trade-off."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_acquisition_optimizer(scenario: Scenario) -> AbstractAcquisitionOptimizer:
        """Returns the acquisition optimizer instance to be used in the BO loop,
        specifying how the acquisition function instance is optimized."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_intensifier(scenario: Scenario) -> AbstractIntensifier:
        """Returns the intensifier instance to be used in the BO loop,
        specifying how to challenge the incumbent configuration on other problem instances."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_initial_design(scenario: Scenario) -> AbstractInitialDesign:
        """Returns an instance of the initial design class to be used in the BO loop,
        specifying how the configurations the BO loop is 'warm-started' with are selected."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_random_design(scenario: Scenario) -> AbstractRandomDesign:
        """Returns an instance of the random design class to be used in the BO loop,
        specifying how to interleave the BO iterations with randomly selected configurations."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_runhistory_encoder(scenario: Scenario) -> AbstractRunHistoryEncoder:
        """Returns an instance of the runhistory encoder class to be used in the BO loop,
        specifying how the runhistory is to be prepared for the next surrogate model."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_multi_objective_algorithm(scenario: Scenario) -> AbstractMultiObjectiveAlgorithm:
        """Returns the multi-objective algorithm instance to be used in the BO loop,
        specifying the scalarization strategy for multiple objectives' costs"""
        raise NotImplementedError

    def _get_optimizer(self) -> SMBO:
        """
        Filling the SMBO with all the pre-initialized components.

        Attributes
        ----------
        optimizer: SMBO
            A fully configured SMBO object
        """
        return SMBO(
            scenario=self._scenario,
            stats=self._stats,
            runner=self._runner,
            initial_design=self._initial_design,
            runhistory=self._runhistory,
            runhistory_encoder=self._runhistory_encoder,
            intensifier=self._intensifier,
            model=self._model,
            acquisition_function=self._acquisition_function,
            acquisition_optimizer=self._acquisition_optimizer,
            random_design=self._random_design,
            overwrite=self._overwrite,
        )

    def _update_dependencies(self) -> None:
        """Convenience method to add some more dependencies. And ensure separable instantiation of
        the components. This is the easiest way to incorporate dependencies, although
        it might be a bit hacky.
        """
        self._intensifier.stats = self._stats
        self._runhistory_encoder._set_multi_objective_algorithm(self._multi_objective_algorithm)
        self._acquisition_function._set_model(self._model)
        self._acquisition_optimizer._set_acquisition_function(self._acquisition_function)
        # TODO: self._runhistory_encoder.set_success_states etc. for different intensifier?

    def _validate(self) -> None:
        """Checks if the composition is correct if there are dependencies, not necessarily"""
        assert self._optimizer

        # Make sure the same acquisition function is used
        assert self._acquisition_function == self._acquisition_optimizer.acquisition_function
