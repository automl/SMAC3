from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("wandb")

from ConfigSpace import ConfigurationSpace, Float
import json
from types import SimpleNamespace

from smac.callback.wandb_callback import WandbCallback
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runner.wandb_runner import WandbRunner


def _mock_wandb_run() -> tuple[MagicMock, MagicMock]:
    """Create mocks for wandb.init and the W&B run used by the tests."""
    mock_init = MagicMock()
    mock_run = MagicMock()
    mock_run.id = "mock_run_id"

    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_run
    mock_context.__exit__.return_value = False
    mock_init.return_value = mock_context

    return mock_init, mock_run


def _scenario(
    objectives: str | list[str] = "cost",
    **kwargs,
) -> Scenario:
    cs = ConfigurationSpace()
    cs.add(Float("x", (0.0, 1.0)))

    return Scenario(
        configspace=cs,
        objectives=objectives,
        **kwargs,
    )


def test_single_objective_result_is_logged() -> None:
    """Test that a single-objective result is logged to W&B under the objective name."""
    scenario = _scenario()

    target_function = lambda config: 0.42

    runner = WandbRunner(
        scenario=scenario,
        target_function=target_function,
        entity="test-entity",
        project="test-project",
        sweep_id="test-sweep",
    )

    config = scenario.configspace.get_default_configuration()
    mock_init, mock_run = _mock_wandb_run()

    with patch("smac.runner.wandb_runner.wandb.init", mock_init):
        result = runner(
            config=config,
            algorithm=target_function,
            algorithm_kwargs={},
        )

    assert result == (0.42, {})
    mock_run.log.assert_called_once_with({"cost": 0.42})


def test_multi_objective_result_is_logged() -> None:
    """Test that multiple named objectives are logged individually to W&B."""
    scenario = _scenario(objectives=["error", "runtime"])

    target_function = lambda config: {"error": 0.4, "runtime": 10.0}

    runner = WandbRunner(
        scenario=scenario,
        target_function=target_function,
        entity="test-entity",
        project="test-project",
        sweep_id="test-sweep",
    )

    config = scenario.configspace.get_default_configuration()
    mock_init, mock_run = _mock_wandb_run()

    with patch("smac.runner.wandb_runner.wandb.init", mock_init):
        result = runner(
            config=config,
            algorithm=target_function,
            algorithm_kwargs={},
        )

    assert result == (
        {"error": 0.4, "runtime": 10.0}, {}
    )
    mock_run.log.assert_called_once_with(
        {"error": 0.4, "runtime": 10.0}
    )


def test_smac_metadata_is_logged(tmp_path) -> None:
    """Test that SMAC-specific trial metadata is added to the W&B run."""
    scenario = _scenario(output_directory=tmp_path)

    runner = WandbRunner(
        scenario=scenario,
        target_function=lambda config: 0.42,
        entity="test-entity",
        project="test-project",
        sweep_id="test-sweep",
    )

    callback = WandbCallback(runner=runner)
    callback._state_path = tmp_path / "wandb_state.json"

    config = scenario.configspace.get_default_configuration()

    # Mock the SMAC RunHistory and the W&B trial run.
    mock_smbo = SimpleNamespace(runhistory=MagicMock())
    mock_smbo.runhistory.get_config_id.return_value = 3

    mock_run = MagicMock()

    info = SimpleNamespace(
        config=config,
        instance="instance-1",
        seed=42,
        budget=10,
    )
    value = SimpleNamespace(additional_info={"wandb_run_id": "mock-run-id"})

    with patch.object(runner, "get_wandb_run", return_value=mock_run):
        callback.on_tell_end(
            smbo=mock_smbo,
            info=info,
            value=value,
        )

    mock_smbo.runhistory.get_config_id.assert_called_once_with(config)

    logged_config = mock_run.config.update.call_args.args[0]

    assert logged_config == {
        "smac/config_id": 3,
        "smac/instance": "instance-1",
        "smac/seed": 42,
        "smac/budget": 10,
    }

def test_runner_works_with_hpo_facade(tmp_path) -> None:
    """Test that WandbRunner integrates correctly with HyperparameterOptimizationFacade."""
    scenario = _scenario(n_trials=2, output_directory=tmp_path)

    runner = WandbRunner(
        scenario=scenario,
        target_function=lambda config: config["x"] ** 2,
        entity="test-entity",
        project="test-project",
        sweep_id="test-sweep",
    )

    mock_init, _ = _mock_wandb_run()
    renamed_runs = [MagicMock(), MagicMock()]

    with (
        patch("smac.runner.wandb_runner.wandb.init", mock_init),
        patch.object(
            runner,
            "get_wandb_run",
            side_effect=renamed_runs,
        ) as mock_get_run,
    ):
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=runner,
        )

        incumbent = smac.optimize()

    assert incumbent is not None
    assert mock_get_run.call_count == 2
    assert renamed_runs[0].name == "test-sweep-trial-0"
    assert renamed_runs[1].name == "test-sweep-trial-1"

def test_callback_initializes_and_persists_state(tmp_path) -> None:
    """Test that a fresh optimization initializes and persists W&B state."""
    scenario = _scenario(output_directory=tmp_path)
    runner = WandbRunner(
        scenario=scenario,
        target_function=lambda config: 0.42,
        entity="test-entity",
        project="test-project",
    )
    callback = WandbCallback(
        runner=runner,
        logging_function=lambda smbo: {"n_evaluations": 0.0},
    )

    with (
        patch(
            "smac.callback.wandb_callback.wandb.sweep",
            return_value="created-sweep",
        ) as mock_sweep,
        patch(
            "smac.callback.wandb_callback.wandb.init",
            return_value=MagicMock(),
        ) as mock_init,
    ):
        callback.on_init(SimpleNamespace(is_resumed=False))

    state_path = scenario.output_directory / "wandb_state.json"
    state = json.loads(state_path.read_text())

    assert runner.sweep_id == "created-sweep"
    assert state["sweep_id"] == "created-sweep"
    assert state["entity"] == "test-entity"
    assert state["project"] == "test-project"
    assert state["observer_run_id"] is not None
    assert state["next_trial_number"] == 0

    mock_sweep.assert_called_once()
    assert mock_init.call_args.kwargs["id"] == state["observer_run_id"]

def test_callback_restores_state_on_resume(tmp_path) -> None:
    """Test that W&B state is restored when resuming an optimization."""
    scenario = _scenario(output_directory=tmp_path)
    state_path = scenario.output_directory / "wandb_state.json"
    state_path.write_text(
        json.dumps(
            {
                "entity": "test-entity",
                "project": "test-project",
                "sweep_id": "existing-sweep",
                "observer_run_id": "existing-observer",
                "next_trial_number": 7,
            }
        )
    )

    runner = WandbRunner(
        scenario=scenario,
        target_function=lambda config: 0.42,
        entity="test-entity",
        project="test-project",
    )
    callback = WandbCallback(
        runner=runner,
        logging_function=lambda smbo: {"n_evaluations": 7.0},
    )

    with (
        patch.object(callback, "_delete_running_runs") as mock_delete,
        patch(
            "smac.callback.wandb_callback.wandb.sweep"
        ) as mock_sweep,
        patch(
            "smac.callback.wandb_callback.wandb.init",
            return_value=MagicMock(),
        ) as mock_init,
    ):
        callback.on_init(SimpleNamespace(is_resumed=True))

    assert runner.sweep_id == "existing-sweep"
    assert callback._observer_run_id == "existing-observer"
    assert callback._trial_counter == 7

    mock_delete.assert_called_once()
    mock_sweep.assert_not_called()
    mock_init.assert_called_once()

    assert mock_init.call_args.kwargs["id"] == "existing-observer"