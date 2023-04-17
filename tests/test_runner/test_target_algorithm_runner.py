from __future__ import annotations

from typing import Callable

import time

import pytest
from ConfigSpace import Configuration, ConfigurationSpace

from smac.runhistory import TrialInfo
from smac.runner.abstract_runner import StatusType
from smac.runner.target_function_runner import TargetFunctionRunner
from smac.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def target(x: float, seed: int, instance: str) -> tuple[float, dict]:
    """Simple target function"""
    return x**2, {"key": seed, "instance": instance}


def target_delayed(x: float, seed: int, instance: str) -> tuple[float, dict]:
    """Target function which sleeps for a second"""
    time.sleep(1)
    return x**2, {"key": seed, "instance": instance}


def target_failed(x: float, seed: int, instance: str) -> tuple[float, dict]:
    """Target function which fails"""
    raise RuntimeError("Failed.")


def target_dummy(config: Configuration, seed: int) -> int:
    """Target function just returning the seed"""
    return seed


def target_multi_objective1(
    config: Configuration,
    seed: int,
    # instance: str,
    # budget: float,
) -> list[float]:
    """Target function multi-objective return the seed twice"""
    return [seed, seed]


def target_multi_objective2(
    config: Configuration,
    seed: int,
    # instance: str,
    # budget: float,
) -> dict[str, float]:
    """Target function multi-objective return the seed twice as a dict"""
    return {"cost1": seed, "cost2": seed}


@pytest.fixture
def make_runner(
    configspace_small: ConfigurationSpace,
    make_scenario: Callable[..., Scenario],
) -> Callable[..., TargetFunctionRunner]:
    """Make a TargetFunctionRunner, ``make_dummy_ta(func)``"""

    def _make(
        target_function: Callable,
        use_multi_objective: bool = False,
        use_instances: bool = False,
    ) -> TargetFunctionRunner:
        scenario = make_scenario(
            configspace=configspace_small,
            use_multi_objective=use_multi_objective,
        )

        required_arguments = ["seed"]
        if use_instances:
            required_arguments += ["instance"]

        return TargetFunctionRunner(
            target_function=target_function,
            scenario=scenario,
            required_arguments=required_arguments,
        )

    return _make


def test_run(make_runner: Callable[..., TargetFunctionRunner]) -> None:
    """Makes sure that we are able to run a config and return the expected values/types"""
    runner = make_runner(target, use_instances=True)
    run_info = TrialInfo(config=2, instance="test", seed=0, budget=0.0)

    # submit runs! then get the value
    runner.submit_trial(run_info)
    result = next(runner.iter_results(), None)

    assert result is not None

    run_info, run_value = result

    assert run_value.cost == 4
    assert run_value.status == StatusType.SUCCESS


def test_serial_runs(make_runner: Callable[..., TargetFunctionRunner]) -> None:
    """Test submitting two runs in succession and that they complete after eachother in results"""
    runner = make_runner(target_delayed, use_instances=True)

    run_info = TrialInfo(config=2, instance="test2", seed=0, budget=0.0)
    runner.submit_trial(run_info)

    run_info = TrialInfo(config=3, instance="test3", seed=0, budget=0.0)
    runner.submit_trial(run_info)

    results = runner.iter_results()

    first = next(results, None)
    assert first is not None

    second = next(results, None)
    assert second is not None

    # To make sure runs launched serially, we just make sure that the end time of a run
    # is later than the other # Results are returned in left to right
    _, first_run_value = first
    _, second_run_value = second
    assert int(first_run_value.endtime) <= int(second_run_value.starttime)


def test_fail(make_runner: Callable[..., TargetFunctionRunner]) -> None:
    """Test traceback and error end up in the additional info of a failing run"""
    runner = make_runner(target_failed, use_instances=True)
    run_info = TrialInfo(config=2, instance="test", seed=0, budget=0.0)

    runner.submit_trial(run_info)
    run_info, run_value = next(runner.iter_results())

    # Make sure the traceback message is included
    assert "traceback" in run_value.additional_info
    assert "RuntimeError" in run_value.additional_info["traceback"]


def test_call(make_runner: Callable[..., TargetFunctionRunner]) -> None:
    """Test call functionality returns things as expected"""
    runner = make_runner(target_dummy)
    config = runner._scenario.configspace.get_default_configuration()

    SEED = 2345
    status, cost, _, _ = runner.run(config=config, instance=None, seed=SEED, budget=None)

    assert cost == SEED
    assert status == StatusType.SUCCESS


def test_multi_objective(make_runner: Callable[..., TargetFunctionRunner]) -> None:
    """Test multiobjective function processed properly"""
    # We always expect a list of costs (although a dict is returned).
    # Internally, target function runner maps the dict to a list of costs in the right order.
    for target in [target_multi_objective1, target_multi_objective2]:
        runner = make_runner(target, use_multi_objective=True)
        config = runner._scenario.configspace.get_default_configuration()

        SEED = 2345
        status, cost, _, _ = runner.run(config=config, instance=None, seed=SEED, budget=None)

        assert isinstance(cost, list)
        assert cost == [SEED, SEED]
        assert status == StatusType.SUCCESS
