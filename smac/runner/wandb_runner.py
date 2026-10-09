from __future__ import annotations

from typing import Any, Callable

import json
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
        A list of arguments passed to the target function by SMAC.
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

        if isinstance(scenario.objectives, str):
            self._objectives = [scenario.objectives]
        else:
            self._objectives = list(scenario.objectives)

        self._sweep_id = sweep_id

    def on_start(self, resumed: bool) -> None:
        """Initializes the W&B Sweep for the optimization.

        Reuses the stored Sweep when resuming an optimization and creates a new
        Sweep otherwise.

        Parameters
        ----------
        resumed : bool
            Whether the optimization is continued from a previous run.
        """
        sweep_file = self._scenario.output_directory / "wandb_sweep.json"

        if resumed and self._sweep_id is None:
            if not sweep_file.exists():
                raise RuntimeError(
                    f"SMAC is continuing a previous run, but no W&B Sweep ID was found at `{sweep_file}`. "
                    "Provide `sweep_id` explicitly to continue this optimization."
                )
            with sweep_file.open() as f:
                self._sweep_id = json.load(f)["sweep_id"]

            self._delete_running_runs()
            return

        self._initialize_sweep_id()

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
            raise RuntimeError(
                "WandbRunner has not been initialized. on_start() must be called before running a trial."
            )

        smac_config = {}
        if "seed" in self._required_arguments:
            smac_config["smac/seed"] = algorithm_kwargs.get("seed")
        if "instance" in self._required_arguments:
            smac_config["smac/instance"] = algorithm_kwargs.get("instance")
        if "budget" in self._required_arguments:
            smac_config["smac/budget"] = algorithm_kwargs.get("budget")

        wandb_config = {**dict(config), **smac_config}
        with wandb.init(
            entity=self._entity,
            project=self._project,
            config=wandb_config,
            settings=wandb.Settings(sweep_id=self._sweep_id),
            reinit="create_new",
        ) as run:
            result = algorithm(config, **algorithm_kwargs)

            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                objective_result, additional_info = result
            else:
                objective_result = result
                additional_info = {}

            run.log(self._result_for_wandb(objective_result))

            return objective_result, additional_info

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

    def _initialize_sweep_id(self) -> None:
        """Initializes and stores the W&B Sweep ID.

        Uses an explicitly provided Sweep ID if available, otherwise creates a
        new W&B Sweep.
        """
        sweep_config: dict[str, Any] = {
            "method": "random",  # W&B is only used for tracking/grouping.
            "parameters": self._parameters_from_configspace(),
        }
        if len(self._objectives) == 1:
            sweep_config["metric"] = {
                "name": self._objectives[0],
                "goal": "minimize",
            }

        if self._sweep_id is None:
            self._sweep_id = wandb.sweep(sweep=sweep_config, entity=self._entity, project=self._project)

        sweep_file = self._scenario.output_directory / "wandb_sweep.json"
        self._scenario.output_directory.mkdir(parents=True, exist_ok=True)

        with sweep_file.open("w") as f:
            json.dump({"sweep_id": self._sweep_id}, f, indent=2)

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

    def _delete_running_runs(self) -> None:
        """Deletes unfinished runs from the current W&B Sweep.

        Running W&B runs may remain after an interrupted SMAC optimization and
        cannot be continued by SMAC after restoring its state.
        """
        api = wandb.Api()
        sweep = api.sweep(f"{self._entity}/{self._project}/{self._sweep_id}")
        for run in sweep.runs:
            if run.state == "running":
                run.delete()
