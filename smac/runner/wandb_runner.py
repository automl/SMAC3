from __future__ import annotations

from typing import Any, Callable

from numbers import Real

import wandb
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)

from smac.callback.callback import Callback
from smac.callback.wandb_callback import WandbCallback
from smac.runhistory import TrialInfo, TrialValue
from smac.runner.target_function_runner import TargetFunctionRunner
from smac.scenario import Scenario


class WandbRunner(TargetFunctionRunner):
    """Target function runner that logs SMAC trials to Weights & Biases.

    Each target-function evaluation is recorded as a separate W&B run and
    associated with a W&B Sweep. SMAC remains responsible for selecting
    configurations and optimizing the objective(s).

    Parameters
    ----------
    scenario : Scenario
    target_function : Callable
        Target function to evaluate.
    entity : str
        W&B entity under which the runs are logged.
    project : str
        W&B project under which the runs are logged.
    sweep_id : str | None, defaults to None
        Existing W&B Sweep ID. If ``None``, a new Sweep is created.
    required_arguments : list[str] | None, defaults to None
        Additional arguments passed to the target function, e.g., seed, instance, budget, or cutoff.
        These must be specified explicitly if required.
    """

    def __init__(
        self,
        scenario: Scenario,
        target_function: Callable,
        *,
        entity: str,
        project: str,
        sweep_id: str | None = None,
        required_arguments: list[str] | None = None,
    ):
        super().__init__(
            scenario=scenario,
            target_function=target_function,
            required_arguments=required_arguments,
        )

        self._scenario = scenario
        self._entity = entity
        self._project = project
        self._trial_number_width = len(str(scenario.n_trials)) if scenario.n_trials is not None else 4

        if isinstance(scenario.objectives, str):
            self._objectives = [scenario.objectives]
        else:
            self._objectives = list(scenario.objectives)

        self._sweep_id = sweep_id

    @property
    def sweep_id(self) -> str | None:
        """Return the W&B Sweep ID associated with this runner."""
        return self._sweep_id

    def set_sweep_id(self, sweep_id: str) -> None:
        """Set the W&B Sweep ID used to associate trial runs.
        The ID is typically initialized or restored by WandbCallback.

        Parameters
        ----------
        sweep_id : str
            ID of the W&B Sweep to associate with this runner.
        """
        if self._sweep_id is not None and self._sweep_id != sweep_id:
            raise ValueError(
                f"W&B Sweep ID mismatch: runner has {self._sweep_id!r}, " f"but callback supplied {sweep_id!r}."
            )

        self._sweep_id = sweep_id

    def __call__(
        self, config: Configuration, algorithm: Callable, algorithm_kwargs: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Calls the target function and logs the result to W&B.

        Parameters
        ----------
        config : Configuration
            Configuration to be passed to the target function.
        algorithm : Callable
            Target function to evaluate.
        algorithm_kwargs : dict[str, Any]
            Additional arguments to be passed to the target function.

        Returns
        -------
        result : float | list[float] | dict[str, float]
            Resulting objective value(s) of the target function.
        additional_info : dict[str, Any]
            Additional information returned by the target function.
        """
        if self._sweep_id is None:
            raise RuntimeError("WandbRunner has not been initialized. on_init() must be called before running a trial.")

        wandb_config = dict(config)
        with wandb.init(
            entity=self._entity,
            project=self._project,
            config=wandb_config,
            settings=wandb.Settings(sweep_id=self._sweep_id),
            reinit="create_new",
        ) as run:
            self._current_trial_run_id = run.id
            result = algorithm(config, **algorithm_kwargs)

            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                objective_result, additional_info = result
            else:
                objective_result = result
                additional_info = {}

            metrics = self._result_for_wandb(objective_result)

            # Prevent additional information from overwriting objective values.
            overlapping_keys = metrics.keys() & additional_info.keys()
            if overlapping_keys:
                raise ValueError(
                    f"Additional information contains keys that conflict with "
                    f"objective metrics: {sorted(overlapping_keys)}"
                )

            # Log objectives and custom trial metrics together.
            run.log({**metrics, **additional_info})

            return objective_result, additional_info

    def run_wrapper(
        self,
        trial_info: TrialInfo,
        **dask_data_to_scatter: dict[str, Any],
    ) -> tuple[TrialInfo, TrialValue]:
        """Execute a trial and preserve its W&B run ID.

        Parameters
        ----------
        trial_info : TrialInfo
            Information describing the trial.
        **dask_data_to_scatter : dict[str, Any]
            Data to scatter to Dask workers.

        Returns
        -------
        info : TrialInfo
            Trial information.
        value : TrialValue
            Trial result and additional information.
        """
        self._current_trial_run_id = None

        info, value = super().run_wrapper(
            trial_info,
            **dask_data_to_scatter,
        )

        if self._current_trial_run_id is not None:
            value.additional_info["wandb_run_id"] = self._current_trial_run_id

        return info, value

    def _result_for_wandb(self, result: Any) -> dict[str, float]:
        """Converts target-function results into W&B metrics.

        Parameters
        ----------
        result : Any
            Result returned by the target function.

        Returns
        -------
        dict[str, float]
            Objective values mapped to their names for logging to W&B.
        """
        if isinstance(result, dict):
            return result

        if isinstance(result, list):
            if len(result) != len(self._objectives):
                raise ValueError(
                    f"Target function returned {len(result)} objective values, "
                    f"but the scenario defines {len(self._objectives)} objectives"
                )
            return {objective: value for objective, value in zip(self._objectives, result)}

        if isinstance(result, Real):
            if len(self._objectives) != 1:
                raise ValueError(
                    f"A scalar objective value was returned, but the scenario "
                    f"defines multiple objectives: {self._objectives}"
                )
            return {
                self._objectives[0]: float(result),
            }

        raise TypeError(f"Unsupported target-function result type: {type(result).__name__}")

    def _parameters_from_configspace(self) -> dict[str, dict[str, Any]]:
        """Converts the SMAC configuration space to W&B sweep parameters.

        The resulting parameters are used as descriptive metadata for the
        W&B Sweep. SMAC remains the source of truth for configuration selection
        and the actual hyperparameter distributions.

        Returns
        -------
        dict[str, dict[str, Any]]
            W&B-formatted parameter definitions.
        """
        parameters: dict[str, dict[str, Any]] = {}

        for hp in self._scenario.configspace.values():
            if isinstance(hp, CategoricalHyperparameter):
                parameters[hp.name] = {"values": list(hp.choices)}
            elif isinstance(hp, OrdinalHyperparameter):
                parameters[hp.name] = {"values": list(hp.sequence)}
            elif isinstance(hp, Constant):
                parameters[hp.name] = {"value": hp.value}
            elif isinstance(hp, (IntegerHyperparameter, FloatHyperparameter)):
                # W&B only provides a descriptive view of the search space here.
                # SMAC remains the source of truth for the actual distribution.
                parameters[hp.name] = {"min": hp.lower, "max": hp.upper}
                if hp.log:
                    parameters[hp.name]["distribution"] = "log_uniform_values"
            else:
                raise ValueError(f"Unsupported ConfigSpace hyperparameter type: {type(hp).__name__} ({hp.name})")

        return parameters

    def get_wandb_run(self, run_id: str) -> wandb.Run:
        """Reopens a W&B run by its ID.

        Returns
        -------
        wandb.Run
            Reopened run.
        """
        return wandb.init(
            entity=self._entity,
            project=self._project,
            id=run_id,
            resume="must",
            reinit="create_new",
        )

    def get_additional_callbacks(
        self,
        existing_callbacks: list[Callback],
    ) -> list[Callback]:
        """Return the required WandbCallback if it is not already registered.

        Parameters
        ----------
        existing_callbacks : list[Callback]
            Callbacks already registered with SMAC.

        Returns
        -------
        list[Callback]
            Additional Callback(s) to register.
        """
        wandb_callbacks = [callback for callback in existing_callbacks if isinstance(callback, WandbCallback)]
        if wandb_callbacks:
            if len(wandb_callbacks) > 1:
                raise ValueError("Only one WandbCallback can be registered.")
            if wandb_callbacks[0]._runner is not self:
                raise ValueError("The supplied WandbCallback must use this WandbRunner.")
            return []

        return [WandbCallback(runner=self)]
