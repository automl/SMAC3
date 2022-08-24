# Add below as a WA for
# https://github.com/dask/distributed/issues/4168
# import multiprocessing.popen_spawn_posix  # noqa
from typing import Callable

import tempfile
import time
import unittest
import unittest.mock

from ConfigSpace import ConfigurationSpace
from dask.distributed import Client

from smac.runhistory import RunInfo, RunValue
from smac.runner.dask_runner import DaskParallelRunner
from smac.runner.runner import StatusType
from smac.runner.target_algorithm_runner import TargetAlgorithmRunner
from smac.scenario import Scenario
from smac.utils.stats import Stats

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def target(x, seed, instance):
    """Simple target function"""
    return x**2, {"key": seed, "instance": instance}


def target_delayed(x, seed, instance):
    """Target function which sleeps for a second"""
    time.sleep(1)
    return x**2, {"key": seed, "instance": instance}


def target_failed(x, seed, instance):
    """Target function which fails"""
    raise RuntimeError("Failed.")


def test_run(
    configspace_small: ConfigurationSpace,
    make_scenario: Callable[..., Scenario],
    make_stats: Callable[..., Stats],
):
    """Makes sure that we are able to run a configuration and get expected values/types"""
    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)

    runner = DaskParallelRunner(
        single_worker=TargetAlgorithmRunner(
            target_algorithm=target,
            scenario=scenario,
            stats=stats,
        ),
        n_workers=2,
    )

    run_info = RunInfo(config=2, instance="test", seed=0, budget=0.0)

    # Submit runs! then get the value
    runner.submit_run(run_info)
    run_values = runner.get_finished_runs()

    # Run will not have finished so fast
    assert len(run_values) == 0
    runner.wait()

    run_values = runner.get_finished_runs()
    assert len(run_values) == 1
    assert isinstance(run_values, list)

    run_info, run_value = run_values[0]
    assert isinstance(run_info, RunInfo)
    assert isinstance(run_value, RunValue)

    assert run_value.cost == 4
    assert run_value.status == StatusType.SUCCESS


def test_parallel_runs(configspace_small, make_scenario, make_stats):
    """Make sure because there are 2 workers, the runs are launched close in time"""
    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)

    runner = DaskParallelRunner(
        single_worker=TargetAlgorithmRunner(
            target_algorithm=target_delayed,
            scenario=scenario,
            stats=stats,
        ),
        n_workers=2,
    )

    run_info = RunInfo(config=2, instance="test", seed=0, budget=0.0)
    runner.submit_run(run_info)

    run_info = RunInfo(config=3, instance="test", seed=0, budget=0.0)
    runner.submit_run(run_info)

    # At this stage, we submitted 2 jobs, that are running in remote workers.
    # We have to wait for each one of them to complete. The runner provides a
    # wait() method to do so, yet it can only wait for a single job to be completed.
    # It does internally via dask wait(<list of futures>) so we take wait for the
    # first job to complete, and take it out
    runner.wait()
    run_values = runner.get_finished_runs()

    # To be on the safe side, we don't check for: self.assertEqual(len(run_values), 1)
    # In the ideal world, two runs were launched which take the same time
    # so waiting for one means, the second one is completed. But some overhead
    # mightcause it to be delayed.

    # Above took the first run results and put it on run_values
    # But for this check we need the second run we submitted
    # Again, the runs might take slightly different time depending if we have
    # heterogeneous workers, so calling wait 2 times is a guarantee that 2
    # jobs are finished
    runner.wait()
    run_values.extend(runner.get_finished_runs())
    assert len(run_values) == 2

    # To make it is parallel, we just make sure that the start of the second run is
    # earlier than the end of the first run
    _, first_run_value = run_values[0]
    _, second_run_value = run_values[1]

    assert int(first_run_value.starttime) <= int(second_run_value.endtime)


def test_additional_info_crash_msg(
    configspace_small: ConfigurationSpace,
    make_scenario: Callable[..., Scenario],
    make_stats: Callable[..., Stats],
) -> None:
    """
    We want to make sure we catch errors as additional info, and in particular when
    doing multiprocessing runs, we want to make sure we capture dask exceptions
    """
    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)

    runner = DaskParallelRunner(
        single_worker=TargetAlgorithmRunner(
            target_algorithm=target_failed,
            scenario=scenario,
            stats=stats,
        ),
        n_workers=2,
    )

    run_info = RunInfo(config=2, instance="test", seed=0, budget=0.0)
    runner.submit_run(run_info)
    runner.wait()
    run_info, result = runner.get_finished_runs()[0]

    # Make sure the traceback message is included
    assert "traceback" in result.additional_info
    assert "RuntimeError" in result.additional_info["traceback"]


def test_internally_created_client() -> None:
    """
    Expects
    -------
    * When no Client is passed, we create one and as such we need to make sure that
      we have a scheduler file is create and the client is closed at the end
    """
    tmp_dir = tempfile.mkdtemp()
    single_worker_mock = unittest.mock.Mock()

    runner = DaskParallelRunner(
        single_worker=single_worker_mock,
        n_workers=1,
        output_directory=tmp_dir,
    )
    assert runner._scheduler_file.exists()

    # Check the client is running
    client = runner._client
    assert client.status == "running"

    # End the runner
    del runner

    # Check the client it created is closed
    assert client.status == "closed"


def test_with_external_client() -> None:
    """
    Expects
    -------
    * A user Client passed directly to a DaskParallelRunner will not not be closed
      upon completion.
    * It will also not create an additional scheduler file
    """
    tmp_dir = tempfile.mkdtemp()

    single_worker_mock = unittest.mock.Mock()
    client = Client()

    # We use the same client twice just to be sure
    for _ in range(2):
        parallel_runner = DaskParallelRunner(
            single_worker=single_worker_mock,
            dask_client=client,
            n_workers=1,
            output_directory=tmp_dir,
        )
        scheduler_file = parallel_runner._scheduler_file
        del parallel_runner

        assert scheduler_file is None
        assert client.status == "running"

    assert client.status == "running"
