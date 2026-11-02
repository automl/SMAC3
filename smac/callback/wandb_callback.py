from __future__ import annotations

from typing import TYPE_CHECKING, Any

import json
import uuid
from collections.abc import Callable
from pathlib import Path

import wandb

if TYPE_CHECKING:
    from smac.runner.wandb_runner import WandbRunner

from smac.callback import Callback
from smac.main.smbo import SMBO
from smac.runhistory import TrialInfo, TrialValue


class WandbCallback(Callback):
    """Log optimization metrics and organize trial runs in W&B.

    Parameters
    ----------
    runner : WandbRunner
        Runner used to log individual trial evaluations.
    logging_function : Callable[[SMBO], dict[str, Any]] | None, defaults to None
        Function returning optimization-level metrics to log after each trial.
        If None, no optimization-level metrics are logged.
    """

    def __init__(
        self,
        runner: WandbRunner,
        logging_function: Callable[[SMBO], dict[str, Any]] | None = None,
    ) -> None:
        self._runner = runner
        self._logging_function = logging_function

        self._trial_counter: int = 0
        self._observer_run_id: str | None = None
        self._state_path: Path | None = None
        self._run = None

    def on_init(self, smbo: SMBO) -> None:
        """Initialize or restore the W&B state for the optimization.

        Parameters
        ----------
        smbo : SMBO
            SMAC optimizer.
        """
        self._state_path = self._runner._scenario.output_directory / "wandb_state.json"
        resumed = smbo.is_resumed

        if resumed:
            state = self._load_state()
            stored_sweep_id = state["sweep_id"]

            # An explicitly supplied ID must agree with the saved state.
            if self._runner.sweep_id is not None and self._runner.sweep_id != stored_sweep_id:
                raise ValueError(
                    f"Cannot resume W&B integration: expected Sweep ID "
                    f"{stored_sweep_id!r}, but received "
                    f"{self._runner.sweep_id!r}."
                )

            self._runner.set_sweep_id(stored_sweep_id)
            self._observer_run_id = state.get("observer_run_id")
            self._trial_counter = int(state["next_trial_number"])
            self._delete_running_runs()
        else:
            # Reuse an explicitly provided sweep ID or create a new sweep.
            sweep_id = self._runner.sweep_id or self._create_sweep()
            self._runner.set_sweep_id(sweep_id)

            self._trial_counter = 0
            self._observer_run_id = uuid.uuid4().hex if self._logging_function is not None else None

            self._save_state()

        self._run = None

        if self._logging_function is not None:
            # Handle the case where logging was enabled only after resuming.
            if self._observer_run_id is None:
                self._observer_run_id = uuid.uuid4().hex
                self._save_state()

            self._run = wandb.init(
                entity=self._runner._entity,
                project=self._runner._project,
                id=self._observer_run_id,
                resume="allow",
                group=self._runner.sweep_id,
                config={"smac/sweep_id": self._runner.sweep_id},
                name=f"{self._runner.sweep_id}-observer",
                job_type="smac_optimization",
                reinit="create_new",
            )

    def _create_sweep(self) -> str:
        """Create a W&B Sweep describing the SMAC configuration space.

        Returns
        -------
        str
            ID of the created W&B Sweep.
        """
        sweep_config: dict[str, Any] = {
            "method": "random",
            "parameters": self._runner._parameters_from_configspace(),
        }

        objectives = self._runner._objectives

        if len(objectives) == 1:
            sweep_config["metric"] = {
                "name": objectives[0],
                "goal": "minimize",
            }

        return wandb.sweep(
            sweep=sweep_config,
            entity=self._runner._entity,
            project=self._runner._project,
        )

    def _delete_running_runs(self) -> None:
        """Deletes unfinished runs from the current W&B Sweep.

        Running W&B runs may remain after an interrupted SMAC optimization and
        cannot be continued by SMAC after restoring its state.
        """
        api = wandb.Api()
        sweep = api.sweep(f"{self._runner._entity}/{self._runner._project}/{self._runner._sweep_id}")
        for run in sweep.runs:
            if run.state == "running":
                run.delete()

    def _save_state(self) -> None:
        """Persist the W&B integration state to the SMAC output directory."""
        if self._state_path is None:
            raise RuntimeError("WandbCallback state path has not been initialized.")

        state = {
            "entity": self._runner._entity,
            "project": self._runner._project,
            "sweep_id": self._runner._sweep_id,
            "observer_run_id": self._observer_run_id,
            "next_trial_number": self._trial_counter,
        }

        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        with self._state_path.open("w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> dict[str, Any]:
        """Load the persisted W&B integration state when resuming an optimization.

        Returns
        -------
        dict[str, Any]
            Previously saved W&B integration state.
        """
        if self._state_path is None or not self._state_path.exists():
            raise RuntimeError(f"Cannot resume W&B integration: callback state was not found at {self._state_path}.")

        with self._state_path.open() as f:
            state = json.load(f)

        expected = {
            "entity": self._runner._entity,
            "project": self._runner._project,
        }

        for key, expected_value in expected.items():
            if state.get(key) != expected_value:
                raise RuntimeError(
                    f"W&B callback state mismatch for {key!r}: stored={state.get(key)!r}, current={expected_value!r}."
                )

        if not state.get("sweep_id"):
            raise RuntimeError("W&B callback state does not contain a Sweep ID.")

        return state

    def on_tell_end(
        self,
        smbo: SMBO,
        info: TrialInfo,
        value: TrialValue,
    ) -> bool | None:
        """Log optimization metrics and update the corresponding trial run.

        Parameters
        ----------
        smbo : SMBO
            SMAC optimizer.
        info : TrialInfo
            Information describing the evaluated trial.
        value : TrialValue
            Result of the evaluated trial.
        """
        if self._logging_function is not None:
            if self._run is None:
                raise RuntimeError(
                    "WandbCallback has not been initialized. " "on_start() must be called before on_tell_end()."
                )

            metrics = self._logging_function(smbo)
            self._run.log(metrics)

        run_id = value.additional_info.get("wandb_run_id")
        if run_id is None:
            raise RuntimeError("No W&B run ID found in trial additional_info.")

        trial_number = self._trial_counter
        self._trial_counter += 1
        self._save_state()

        run = self._runner.get_wandb_run(run_id)

        try:
            # Associate the W&B run with its SMAC trial.
            trial_metadata = {"smac/config_id": smbo.runhistory.get_config_id(info.config)}
            for name in ("instance", "seed", "budget"):
                value_ = getattr(info, name)
                if value_ is not None:
                    trial_metadata[f"smac/{name}"] = value_

            run.config.update(trial_metadata, allow_val_change=True)

            formatted_trial_number = str(trial_number).zfill(self._runner._trial_number_width)
            run.name = f"{self._runner._sweep_id}-trial-{formatted_trial_number}"
        finally:
            run.finish()

        return None

    def on_end(self, smbo: SMBO) -> None:
        """Finish the optimization-level W&B run.

        Parameters
        ----------
        smbo : SMBO
            SMAC optimizer.
        """
        if self._run is not None:
            self._run.finish()
            self._run = None
