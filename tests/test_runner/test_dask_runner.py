from __future__ import annotations

from typing import Callable

import time

import pytest
from ConfigSpace import ConfigurationSpace
from dask.distributed import Client

from smac.runhistory import RunInfo, RunValue
from smac.runner.dask_runner import DaskParallelRunner
from smac.runner.runner import StatusType
from smac.runner.target_algorithm_runner import TargetAlgorithmRunner
from smac.scenario import Scenario
from smac.utils.stats import Stats

# https://github.com/dask/distributed/issues/4168
# import multiprocessing.popen_spawn_posix  # noqa

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


@pytest.fixture
def make_dummy_ta(
    configspace_small: ConfigurationSpace,
    make_scenario: Callable[..., Scenario],
    make_stats: Callable[..., Stats],
) -> Callable[..., TargetAlgorithmRunner]:
    """Make a TargetAlgorithmRunner, ``make_dummy_ta(func)``"""

    def _make(target_algorithm: Callable, n_workers: int = 2) -> TargetAlgorithmRunner:
        scenario = make_scenario(configspace=configspace_small, n_workers=n_workers)
        return TargetAlgorithmRunner(
            target_algorithm=target_algorithm,
            scenario=scenario,
            stats=make_stats(scenario),
        )

    return _make


def test_run(make_dummy_ta: Callable[..., TargetAlgorithmRunner]) -> None:
    """Makes sure that we are able to run a configuration and get expected values/types"""
    single_worker = make_dummy_ta(target, n_workers=2)
    runner = DaskParallelRunner(single_worker=single_worker)
    print(runner.available_worker_count())
    print(len(runner._pending_runs))
    print(sum(runner._client.nthreads().values()))

    run_info = RunInfo(config=2, instance="test", seed=0, budget=0.0)

    # Submit runs! then get the value
    runner.submit_run(run_info)
    result = next(runner.iter_results(), None)

    # Run will not have finished so fast
    assert result is None

    # Wait until there is a result
    runner.wait()
    result = next(runner.iter_results(), None)
    assert result is not None

    run_info, run_value = result
    assert isinstance(run_info, RunInfo)
    assert isinstance(run_value, RunValue)

    assert run_value.cost == 4
    assert run_value.status == StatusType.SUCCESS


def test_parallel_runs(make_dummy_ta: Callable[..., TargetAlgorithmRunner]) -> None:
    """Make sure because there are 2 workers, the runs are launched close in time"""
    single_runner = make_dummy_ta(target_delayed, n_workers=2)
    runner = DaskParallelRunner(single_worker=single_runner)

    assert runner.available_worker_count() == 2

    run_info = RunInfo(config=2, instance="test", seed=0, budget=0.0)
    runner.submit_run(run_info)

    assert runner.available_worker_count() == 1

    run_info = RunInfo(config=3, instance="test", seed=0, budget=0.0)
    runner.submit_run(run_info)

    assert runner.available_worker_count() == 0

    # At this stage, we submitted 2 jobs, that are running in remote workers.
    # We have to wait for each one of them to complete. The runner provides a
    # wait() method to do so, yet it can only wait for a single job to be completed.
    # It does internally via dask wait(<list of futures>) so we take wait for the
    # first job to complete, and take it out
    runner.wait()
    first = next(runner.iter_results(), None)

    # The iter results could have freed up two so we only check it's one or greater
    assert first is not None
    assert runner.available_worker_count() >= 1

    # Again, the runs might take slightly different time depending if we have
    # heterogeneous workers, so calling wait 2 times is a guarantee that 2
    # jobs are finished
    runner.wait()
    second = next(runner.iter_results(), None)

    assert second is not None
    assert runner.available_worker_count() == 2

    # To make it is parallel, we just make sure that the start of the second run is
    # earlier than the end of the first run
    _, first_run_value = first
    _, second_run_value = second

    assert int(first_run_value.starttime) <= int(second_run_value.endtime)


def test_additional_info_crash_msg(make_dummy_ta: Callable[..., TargetAlgorithmRunner]) -> None:
    """
    We want to make sure we catch errors as additional info, and in particular when
    doing multiprocessing runs, we want to make sure we capture dask exceptions
    """
    single_worker = make_dummy_ta(target_failed, n_workers=2)
    runner = DaskParallelRunner(single_worker=single_worker)

    run_info = RunInfo(config=2, instance="test", seed=0, budget=0.0)
    runner.submit_run(run_info)
    runner.wait()
    run_info, run_value = next(runner.iter_results())

    # Make sure the traceback message is included
    assert "traceback" in run_value.additional_info
    assert "RuntimeError" in run_value.additional_info["traceback"]


def test_internally_created_client(make_dummy_ta: Callable[..., TargetAlgorithmRunner]) -> None:
    """
    Expects
    -------
    * When no Client is passed, we create one and as such we need to make sure that
      we have a scheduler file is create and the client is closed at the end
    """
    single_worker = make_dummy_ta(target, n_workers=2)
    runner = DaskParallelRunner(single_worker=single_worker)

    runner = DaskParallelRunner(single_worker=runner)
    assert runner._scheduler_file is not None and runner._scheduler_file.exists()

    # Check the client is running
    client = runner._client
    assert client.status == "running"

    # End the runner
    runner.close()

    # Check the client it created is closed
    assert client.status == "closed"


def test_with_external_client(make_dummy_ta: Callable[..., TargetAlgorithmRunner]) -> None:
    """
    Expects
    -------
    * A user Client passed directly to a DaskParallelRunner will not not be closed
      upon completion.
    * It will also not create an additional scheduler file
    """
    client = Client()

    # We use the same client twice just to be sure
    for _ in range(2):
        single_worker = make_dummy_ta(target, n_workers=2)
        runner = DaskParallelRunner(
            single_worker=single_worker,
            dask_client=client,
        )

        # Calling close should not do anything to
        # the client
        runner.close()

        assert runner._scheduler_file is None
        assert client.status == "running"

    assert client.status == "running"
    client.close()
