from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("wandb")

from ConfigSpace import ConfigurationSpace, Float

from smac import HyperparameterOptimizationFacade, Scenario
from smac.runner.wandb_runner import WandbRunner


def _mock_wandb_run() -> tuple[MagicMock, MagicMock]:
    """Create mocks for wandb.init and the W&B run used by the tests."""
    mock_init = MagicMock()
    mock_run = MagicMock()

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
    scenario = _scenario(
        objectives=["error", "runtime"],
    )

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
        {
            "error": 0.4,
            "runtime": 10.0,
        },
        {},
    )
    mock_run.log.assert_called_once_with(
        {
            "error": 0.4,
            "runtime": 10.0,
        }
    )


def test_smac_metadata_is_logged() -> None:
    """Test that SMAC-specific trial metadata is added to the W&B run configuration."""
    scenario = _scenario()

    def target_function(config, instance, budget):
        return 0.42

    runner = WandbRunner(
        scenario=scenario,
        target_function=target_function,
        entity="test-entity",
        project="test-project",
        sweep_id="test-sweep",
        required_arguments=["instance", "budget"],
    )

    config = scenario.configspace.get_default_configuration()
    mock_init, _ = _mock_wandb_run()

    with patch("smac.runner.wandb_runner.wandb.init", mock_init):
        runner(
            config=config,
            algorithm=target_function,
            algorithm_kwargs={
                "instance": "instance-1",
                "budget": 10,
            },
        )

    wandb_config = mock_init.call_args.kwargs["config"]

    assert "smac/seed" not in wandb_config
    assert wandb_config["smac/instance"] == "instance-1"
    assert wandb_config["smac/budget"] == 10


def test_runner_works_with_hpo_facade(tmp_path) -> None:
    """Test that WandbRunner integrates correctly with HyperparameterOptimizationFacade."""
    scenario = _scenario(
        n_trials=2,
        output_directory=tmp_path,
    )

    runner = WandbRunner(
        scenario=scenario,
        target_function=lambda config: config["x"] ** 2,
        entity="test-entity",
        project="test-project",
        sweep_id="test-sweep",
    )

    mock_init, _ = _mock_wandb_run()

    with patch("smac.runner.wandb_runner.wandb.init", mock_init):
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=runner,
        )

        incumbent = smac.optimize()

    assert incumbent is not None
    assert mock_init.call_count == 2