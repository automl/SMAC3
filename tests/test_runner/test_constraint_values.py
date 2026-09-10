from __future__ import annotations

from typing import Callable

import pytest
from ConfigSpace import Configuration, ConfigurationSpace

from smac.runhistory import TrialInfo
from smac.runner.abstract_runner import StatusType
from smac.runner.target_function_runner import TargetFunctionRunner
from smac.scenario import Scenario

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


def target_with_constraint(config: Configuration, seed: int) -> tuple[float, dict]:
    """Target function reporting a constrained output next to unrelated information"""
    return 0.5, {"latency": 93.2, "note": "kept"}


def target_without_constraint(config: Configuration, seed: int) -> tuple[float, dict]:
    """Target function that never reports the constrained output"""
    return 0.5, {"note": "kept"}


def target_crashing(config: Configuration, seed: int) -> tuple[float, dict]:
    """Target function which fails before it can report anything"""
    raise RuntimeError("Failed.")


@pytest.fixture
def make_runner(configspace_small: ConfigurationSpace) -> Callable[..., TargetFunctionRunner]:
    def _make(target_function: Callable, constraints: list[str] | None) -> TargetFunctionRunner:
        scenario = Scenario(configspace=configspace_small, constraints=constraints, n_trials=10)

        return TargetFunctionRunner(
            target_function=target_function,
            scenario=scenario,
            required_arguments=["seed"],
        )

    return _make


def _run(runner: TargetFunctionRunner, configspace_small: ConfigurationSpace):
    config = configspace_small.sample_configuration()
    runner.submit_trial(TrialInfo(config=config, seed=0))
    _, trial_value = next(runner.iter_results())

    return trial_value


def test_declared_constraint_is_moved_out_of_additional_info(make_runner, configspace_small) -> None:
    """A declared output is reported as a constraint value, not as free-form information.

    Runs a target function returning both a constrained output and an unrelated key, then checks the constrained
    one landed in constraint_values while the other stayed in additional_info.
    """
    trial_value = _run(make_runner(target_with_constraint, ["latency <= 100"]), configspace_small)

    assert trial_value.constraint_values == {"latency": 93.2}
    assert trial_value.additional_info == {"note": "kept"}


def test_undeclared_outputs_are_left_alone(make_runner, configspace_small) -> None:
    """Without a declared constraint nothing is extracted.

    Runs the same target function on an unconstrained scenario and checks the output stayed in additional_info.
    """
    trial_value = _run(make_runner(target_with_constraint, None), configspace_small)

    assert trial_value.constraint_values is None
    assert trial_value.additional_info == {"latency": 93.2, "note": "kept"}


def test_an_unreported_constraint_is_simply_absent(make_runner, configspace_small) -> None:
    """A target function that reports nothing for a constraint does not break the run.

    Runs a target function omitting the constrained output and checks the trial still succeeds with no value
    recorded, which counts as infeasible downstream.
    """
    trial_value = _run(make_runner(target_without_constraint, ["latency <= 100"]), configspace_small)

    assert trial_value.status == StatusType.SUCCESS
    assert trial_value.constraint_values == {}


def test_a_crashed_trial_reports_no_constraint_value(make_runner, configspace_small) -> None:
    """A crash never gets the chance to report a constrained output.

    Runs a failing target function and checks the trial is CRASHED with no constraint value and its traceback
    still intact.
    """
    trial_value = _run(make_runner(target_crashing, ["latency <= 100"]), configspace_small)

    assert trial_value.status == StatusType.CRASHED
    assert trial_value.constraint_values == {}
    assert "traceback" in trial_value.additional_info
