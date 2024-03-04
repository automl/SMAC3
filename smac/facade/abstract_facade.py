from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from pathlib import Path

import joblib
from ConfigSpace import Configuration
from dask.distributed import Client
from typing_extensions import Literal

import smac
from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer.abstract_acqusition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.callback.callback import Callback
from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.main.config_selector import ConfigSelector
from smac.main.smbo import SMBO
from smac.model.abstract_model import AbstractModel
from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runhistory.encoder.abstract_encoder import AbstractRunHistoryEncoder
from smac.runhistory.runhistory import RunHistory
from smac.runner.abstract_runner import AbstractRunner
from smac.runner.dask_runner import DaskParallelRunner
from smac.runner.target_function_runner import TargetFunctionRunner
from smac.runner.target_function_script_runner import TargetFunctionScriptRunner
from smac.scenario import Scenario
from smac.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class AbstractFacade:
    """Facade is an abstraction on top of the SMBO backend to organize the components of a Bayesian Optimization loop
    in a configurable and separable manner to suit the various needs of different (hyperparameter) optimization
    pipelines.

    With the exception to scenario and ``target_function``, which are expected of the user, the parameters ``model``,
    ``acquisition_function``, ``acquisition_maximizer``, ``initial_design``, ``random_design``, ``intensifier``,
    ``multi_objective_algorithm``, ``runhistory_encoder`` can either be explicitly specified in the subclasses'
    ``get_*`` methods (defining a specific BO pipeline) or be instantiated by the user to overwrite a pipeline
    components explicitly.

    Parameters
    ----------
    scenario : Scenario
        The scenario object, holding all environmental information.
    target_function : Callable | str | AbstractRunner
        This function is called internally to judge a trial's performance. If a string is passed,
        it is assumed to be a script. In this case, ``TargetFunctionScriptRunner`` is used to run the script.
    model : AbstractModel | None, defaults to None
        The surrogate model.
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
        The acquisition function.
    acquisition_maximizer : AbstractAcquisitionMaximizer | None, defaults to None
        The acquisition maximizer, deciding which configuration is most promising based on the surrogate model and
        acquisition function.
    initial_design : InitialDesign | None, defaults to None
        The sampled configurations from the initial design are evaluated before the Bayesian optimization loop starts.
    random_design : RandomDesign | None, defaults to None
        The random design is used in the acquisition maximizer, deciding whether the next configuration should be drawn
        from the acquisition function or randomly.
    intensifier : AbstractIntensifier | None, defaults to None
        The intensifier decides which trial (combination of configuration, seed, budget and instance) should be run
        next.
    multi_objective_algorithm : AbstractMultiObjectiveAlgorithm | None, defaults to None
        In case of multiple objectives, the objectives need to be interpreted so that an optimization is possible.
        The multi-objective algorithm takes care of that.
    runhistory_encoder : RunHistoryEncoder | None, defaults to None
        Based on the runhistory, the surrogate model is trained. However, the data first needs to be encoded, which
        is done by the runhistory encoder. For example, inactive hyperparameters need to be encoded or cost values
        can be log transformed.
    logging_level: int | Path | Literal[False] | None
        The level of logging (the lowest level 0 indicates the debug level). If a path is passed, a yaml file is
        expected with the logging configuration. If nothing is passed, the default logging.yml from SMAC is used.
        If False is passed, SMAC will not do any customization of the logging setup and the responsibility is left
        to the user.
    callbacks: list[Callback], defaults to []
        Callbacks, which are incorporated into the optimization loop.
    overwrite: bool, defaults to False
        When True, overwrites the run results if a previous run is found that is
        consistent in the meta data with the current setup. When False and a previous run is found that is
        consistent in the meta data, the run is continued. When False and a previous run is found that is
        not consistent in the meta data, the the user is asked for the exact behaviour (overwrite completely
        or rename old run first).
    dask_client: Client | None, defaults to None
        User-created dask client, which can be used to start a dask cluster and then attach SMAC to it. This will not
        be closed automatically and will have to be closed manually if provided explicitly. If none is provided
        (default), a local one will be created for you and closed upon completion.
    """

    def __init__(
        self,
        scenario: Scenario,
        target_function: Callable | str | AbstractRunner,
        *,
        model: AbstractModel | None = None,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        acquisition_maximizer: AbstractAcquisitionMaximizer | None = None,
        initial_design: AbstractInitialDesign | None = None,
        random_design: AbstractRandomDesign | None = None,
        intensifier: AbstractIntensifier | None = None,
        multi_objective_algorithm: AbstractMultiObjectiveAlgorithm | None = None,
        runhistory_encoder: AbstractRunHistoryEncoder | None = None,
        config_selector: ConfigSelector | None = None,
        logging_level: int | Path | Literal[False] | None = None,
        callbacks: list[Callback] = None,
        overwrite: bool = False,
        dask_client: Client | None = None,
    ):
        setup_logging(logging_level)

        if callbacks is None:
            callbacks = []

        if model is None:
            model = self.get_model(scenario)

        if acquisition_function is None:
            acquisition_function = self.get_acquisition_function(scenario)

        if acquisition_maximizer is None:
            acquisition_maximizer = self.get_acquisition_maximizer(scenario)

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

        if config_selector is None:
            config_selector = self.get_config_selector(scenario)

        # Initialize empty stats and runhistory object
        runhistory = RunHistory(multi_objective_algorithm=multi_objective_algorithm)

        # Set the seed for configuration space
        scenario.configspace.seed(scenario.seed)

        # Set variables globally
        self._scenario = scenario
        self._model = model
        self._acquisition_function = acquisition_function
        self._acquisition_maximizer = acquisition_maximizer
        self._initial_design = initial_design
        self._random_design = random_design
        self._intensifier = intensifier
        self._multi_objective_algorithm = multi_objective_algorithm
        self._runhistory = runhistory
        self._runhistory_encoder = runhistory_encoder
        self._config_selector = config_selector
        self._callbacks = callbacks
        self._overwrite = overwrite

        # Prepare the algorithm executer
        runner: AbstractRunner
        if isinstance(target_function, AbstractRunner):
            runner = target_function
        elif isinstance(target_function, str):
            runner = TargetFunctionScriptRunner(
                scenario=scenario,
                target_function=target_function,
                required_arguments=self._get_signature_arguments(),
            )
        else:
            runner = TargetFunctionRunner(
                scenario=scenario,
                target_function=target_function,
                required_arguments=self._get_signature_arguments(),
            )

        # In case of multiple jobs, we need to wrap the runner again using DaskParallelRunner
        if (n_workers := scenario.n_workers) > 1 or dask_client is not None:
            if dask_client is not None and n_workers > 1:
                logger.warning(
                    "Provided `dask_client`. Ignore `scenario.n_workers`, directly set `n_workers` in `dask_client`."
                )
            else:
                available_workers = joblib.cpu_count()
                if n_workers > available_workers:
                    logger.info(f"Workers are reduced to {n_workers}.")
                    n_workers = available_workers

            # We use a dask runner for parallelization
            runner = DaskParallelRunner(single_worker=runner, dask_client=dask_client)

        # Set the runner to access it globally
        self._runner = runner

        # Adding dependencies of the components
        self._update_dependencies()

        # We have to update our meta data (basically arguments of the components)
        self._scenario._set_meta(self.meta)

        # We have to validate if the object compositions are correct and actually make sense
        self._validate()

        # Finally we configure our optimizer
        self._optimizer = self._get_optimizer()
        assert self._optimizer

        # Register callbacks here
        for callback in callbacks:
            self._optimizer.register_callback(callback)

        # Additionally, we register the runhistory callback from the intensifier to efficiently update our incumbent
        # every time new information are available
        self._optimizer.register_callback(self._intensifier.get_callback(), index=0)

    @property
    def scenario(self) -> Scenario:
        """The scenario object which holds all environment information."""
        return self._scenario

    @property
    def runhistory(self) -> RunHistory:
        """The runhistory which is filled with all trials during the optimization process."""
        return self._optimizer._runhistory

    @property
    def optimizer(self) -> SMBO:
        """The optimizer which is responsible for the BO loop. Keeps track of useful information like status."""
        return self._optimizer

    @property
    def intensifier(self) -> AbstractIntensifier:
        """The optimizer which is responsible for the BO loop. Keeps track of useful information like status."""
        return self._intensifier

    @property
    def meta(self) -> dict[str, Any]:
        """Generates a hash based on all components of the facade. This is used for the run name or to determine
        whether a run should be continued or not.
        """
        multi_objective_algorithm_meta = None
        if self._multi_objective_algorithm is not None:
            multi_objective_algorithm_meta = self._multi_objective_algorithm.meta

        meta = {
            "facade": {"name": self.__class__.__name__},
            "runner": self._runner.meta,
            "model": self._model.meta,
            "acquisition_maximizer": self._acquisition_maximizer.meta,
            "acquisition_function": self._acquisition_function.meta,
            "intensifier": self._intensifier.meta,
            "initial_design": self._initial_design.meta,
            "random_design": self._random_design.meta,
            "runhistory_encoder": self._runhistory_encoder.meta,
            "multi_objective_algorithm": multi_objective_algorithm_meta,
            "config_selector": self._config_selector.meta,
            "version": smac.version,
        }

        return meta

    def ask(self) -> TrialInfo:
        """Asks the intensifier for the next trial."""
        return self._optimizer.ask()

    def tell(self, info: TrialInfo, value: TrialValue, save: bool = True) -> None:
        """Adds the result of a trial to the runhistory and updates the intensifier.

        Parameters
        ----------
        info: TrialInfo
            Describes the trial from which to process the results.
        value: TrialValue
            Contains relevant information regarding the execution of a trial.
        save : bool, optional to True
            Whether the runhistory should be saved.
        """
        return self._optimizer.tell(info, value, save=save)

    def optimize(self, *, data_to_scatter: dict[str, Any] | None = None) -> Configuration | list[Configuration]:
        """
        Optimizes the configuration of the algorithm.

        Parameters
        ----------
        data_to_scatter: dict[str, Any] | None
            We first note that this argument is valid only dask_runner!
            When a user scatters data from their local process to the distributed network,
            this data is distributed in a round-robin fashion grouping by number of cores.
            Roughly speaking, we can keep this data in memory and then we do not have to (de-)serialize the data
            every time we would like to execute a target function with a big dataset.
            For example, when your target function has a big dataset shared across all the target function,
            this argument is very useful.

        Returns
        -------
        incumbent : Configuration
            Best found configuration.
        """
        incumbents = None
        if isinstance(data_to_scatter, dict) and len(data_to_scatter) == 0:
            raise ValueError("data_to_scatter must be None or dict with some elements, but got an empty dict.")

        try:
            incumbents = self._optimizer.optimize(data_to_scatter=data_to_scatter)
        finally:
            self._optimizer.save()

        return incumbents

    def validate(
        self,
        config: Configuration,
        *,
        seed: int | None = None,
    ) -> float | list[float]:
        """Validates a configuration on seeds different from the ones used in the optimization process and on the
        highest budget (if budget type is real-valued).

        Parameters
        ----------
        config : Configuration
            Configuration to validate
        instances : list[str] | None, defaults to None
            Which instances to validate. If None, all instances specified in the scenario are used.
            In case that the budget type is real-valued, this argument is ignored.
        seed : int | None, defaults to None
            If None, the seed from the scenario is used.

        Returns
        -------
        cost : float | list[float]
            The averaged cost of the configuration. In case of multi-fidelity, the cost of each objective is
            averaged.
        """
        return self._optimizer.validate(config, seed=seed)

    @staticmethod
    @abstractmethod
    def get_model(scenario: Scenario) -> AbstractModel:
        """Returns the surrogate cost model instance used in the BO loop."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_acquisition_function(scenario: Scenario) -> AbstractAcquisitionFunction:
        """Returns the acquisition function instance used in the BO loop,
        defining the exploration/exploitation trade-off.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_acquisition_maximizer(scenario: Scenario) -> AbstractAcquisitionMaximizer:
        """Returns the acquisition optimizer instance to be used in the BO loop,
        specifying how the acquisition function instance is optimized.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_intensifier(scenario: Scenario) -> AbstractIntensifier:
        """Returns the intensifier instance to be used in the BO loop,
        specifying how to challenge the incumbent configuration on other problem instances.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_initial_design(scenario: Scenario) -> AbstractInitialDesign:
        """Returns an instance of the initial design class to be used in the BO loop,
        specifying how the configurations the BO loop is 'warm-started' with are selected.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_random_design(scenario: Scenario) -> AbstractRandomDesign:
        """Returns an instance of the random design class to be used in the BO loop,
        specifying how to interleave the BO iterations with randomly selected configurations.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_runhistory_encoder(scenario: Scenario) -> AbstractRunHistoryEncoder:
        """Returns an instance of the runhistory encoder class to be used in the BO loop,
        specifying how the runhistory is to be prepared for the next surrogate model.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_multi_objective_algorithm(scenario: Scenario) -> AbstractMultiObjectiveAlgorithm:
        """Returns the multi-objective algorithm instance to be used in the BO loop,
        specifying the scalarization strategy for multiple objectives' costs.
        """
        raise NotImplementedError

    @staticmethod
    def get_config_selector(
        scenario: Scenario,
        *,
        retrain_after: int = 8,
        retries: int = 16,
    ) -> ConfigSelector:
        """Returns the default configuration selector."""
        return ConfigSelector(scenario, retrain_after=retrain_after, retries=retries)

    def _get_optimizer(self) -> SMBO:
        """Fills the SMBO with all the pre-initialized components."""
        return SMBO(
            scenario=self._scenario,
            runner=self._runner,
            runhistory=self._runhistory,
            intensifier=self._intensifier,
            overwrite=self._overwrite,
        )

    def _update_dependencies(self) -> None:
        """Convenience method to add some more dependencies. And ensure separable instantiation of
        the components. This is the easiest way to incorporate dependencies, although
        it might be a bit hacky.
        """
        # Set components to config selector
        self._config_selector._set_components(
            initial_design=self._initial_design,
            runhistory=self._runhistory,
            runhistory_encoder=self._runhistory_encoder,
            model=self._model,
            acquisition_function=self._acquisition_function,
            acquisition_maximizer=self._acquisition_maximizer,
            random_design=self._random_design,
            callbacks=self._callbacks,
        )

        self._runhistory_encoder.multi_objective_algorithm = self._multi_objective_algorithm
        self._runhistory_encoder.runhistory = self._runhistory
        self._acquisition_function.model = self._model
        self._acquisition_maximizer.acquisition_function = self._acquisition_function
        self._intensifier.config_selector = self._config_selector
        self._intensifier.runhistory = self._runhistory

    def _validate(self) -> None:
        """Checks if the composition is correct if there are dependencies."""
        # Make sure the same acquisition function is used
        assert self._acquisition_function == self._acquisition_maximizer._acquisition_function

        if isinstance(self._runner, DaskParallelRunner) and (
            self.scenario.trial_walltime_limit is not None or self.scenario.trial_memory_limit is not None
        ):
            # This is probably due to pickling dask jobs
            raise ValueError(
                "Parallelization via Dask cannot be used in combination with limiting "
                "the resources "
                "of the target function via `scenario.trial_walltime_limit` or "
                "`scenario.trial_memory_limit`. Set those to `None` if you want "
                "parallelization. "
            )

    def _get_signature_arguments(self) -> list[str]:
        """Returns signature arguments, which are required by the intensifier."""
        arguments = []

        if self._intensifier.uses_seeds:
            arguments += ["seed"]

        if self._intensifier.uses_budgets:
            arguments += ["budget"]

        if self._intensifier.uses_instances:
            arguments += ["instance"]

        return arguments
