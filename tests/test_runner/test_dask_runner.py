# Add below as a WA for
# https://github.com/dask/distributed/issues/4168
# import multiprocessing.popen_spawn_posix  # noqa
import os
import tempfile
import time
import unittest
import unittest.mock

# import dask  # noqa
from dask.distributed import Client

from smac.configspace import ConfigurationSpace
from smac.runhistory import RunInfo, RunValue
from smac.runner.runner import StatusType
from smac.runner.dask_runner import DaskParallelRunner
from smac.runner.target_algorithm_runner import TargetAlgorithmRunner
from smac.scenario import Scenario
from smac.utils.stats import Stats

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def target(x, seed, instance):
    return x**2, {"key": seed, "instance": instance}


def target_delayed(x, seed, instance):
    time.sleep(1)
    return x**2, {"key": seed, "instance": instance}


def target_failed(x, seed, instance):
    raise RuntimeError("Failed.")


def test_run(configspace_small, make_scenario, make_stats):
    """Makes sure that we are able to run a configuration and
    return the expected values/types"""

    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)

    runner = TargetAlgorithmRunner(target_algorithm=target, scenario=scenario, stats=stats)
    runner = DaskParallelRunner(runner, n_workers=2)
    assert isinstance(runner, DaskParallelRunner)

    run_info = RunInfo(
        config=2,
        instance="test",
        seed=0,
        budget=0.0,
    )

    # Submit runs! then get the value
    runner.submit_run(run_info)
    run_values = runner.get_finished_runs()

    # Run will not have finished so fast
    assert len(run_values) == 0
    runner.wait()

    run_values = runner.get_finished_runs()
    assert len(run_values) == 1
    assert isinstance(run_values, list)
    assert isinstance(run_values[0][0], RunInfo)
    assert isinstance(run_values[0][1], RunValue)
    assert run_values[0][1].cost == 4
    assert run_values[0][1].status == StatusType.SUCCESS


def test_parallel_runs(configspace_small, make_scenario, make_stats):
    """Make sure because there are 2 workers, the runs are launched
    closely in time together"""

    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)

    runner = TargetAlgorithmRunner(target_algorithm=target_delayed, scenario=scenario, stats=stats)
    runner = DaskParallelRunner(runner, n_workers=2)
    assert isinstance(runner, DaskParallelRunner)

    run_info = RunInfo(
        config=2,
        instance="test",
        seed=0,
        budget=0.0,
    )
    runner.submit_run(run_info)
    run_info = RunInfo(
        config=3,
        instance="test",
        seed=0,
        budget=0.0,
    )
    runner.submit_run(run_info)

    # At this stage, we submitted 2 jobs, that are running in remote
    # workers. We have to wait for each one of them to complete. The
    # runner provides a wait() method to do so, yet it can only wait for
    # a single job to be completed. It does internally via dask wait(<list of futures>)
    # so we take wait for the first job to complete, and take it out
    runner.wait()
    run_values = runner.get_finished_runs()

    # To be on the safe side, we don't check for:
    # self.assertEqual(len(run_values), 1)
    # In the ideal world, two runs were launched which take the same time
    # so waiting for one means, the second one is completed. But some
    # overhead might cause it to be delayed.

    # Above took the first run results and put it on run_values
    # But for this check we need the second run we submitted
    # Again, the runs might take slightly different time depending if we have
    # heterogeneous workers, so calling wait 2 times is a guarantee that 2
    # jobs are finished
    runner.wait()
    run_values.extend(runner.get_finished_runs())
    assert len(run_values) == 2

    # To make it is parallel, we just make sure that the start of the second
    # run is earlier than the end of the first run
    # Results are returned in left to right
    assert int(run_values[0][1].starttime) <= int(run_values[1][1].endtime)


def test_num_workers(configspace_small, make_scenario, make_stats):
    """Make sure we can properly return the number of workers"""

    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)

    runner = TargetAlgorithmRunner(target_algorithm=target_delayed, scenario=scenario, stats=stats)
    runner = DaskParallelRunner(runner, n_workers=2)
    assert isinstance(runner, DaskParallelRunner)

    # Reduce the number of workers
    # have to give time for the worker to be killed
    runner.client.cluster.scale(1)
    time.sleep(2)
    assert runner.num_workers() == 1


def test_additional_info_crash_msg(configspace_small, make_scenario, make_stats):
    """
    We want to make sure we catch errors as additional info,
    and in particular when doing multiprocessing runs, we
    want to make sure we capture dask exceptions
    """

    scenario = make_scenario(configspace_small)
    stats = make_stats(scenario)

    runner = TargetAlgorithmRunner(target_algorithm=target_failed, scenario=scenario, stats=stats)
    runner = DaskParallelRunner(runner, n_workers=2)
    assert isinstance(runner, DaskParallelRunner)

    run_info = RunInfo(
        config=2,
        instance="test",
        seed=0,
        budget=0.0,
    )
    runner.submit_run(run_info)
    runner.wait()
    run_info, result = runner.get_finished_runs()[0]

    # Make sure the traceback message is included
    assert "traceback" in result.additional_info
    assert "RuntimeError" in result.additional_info["traceback"]


def test_file_output():
    tmp_dir = tempfile.mkdtemp()
    single_worker_mock = unittest.mock.Mock()
    _ = DaskParallelRunner(
        single_worker=single_worker_mock,
        n_workers=1,
        output_directory=tmp_dir,
    )

    assert os.path.exists(os.path.join(tmp_dir, ".dask_scheduler_file"))


def test_do_not_close_external_client():
    tmp_dir = tempfile.mkdtemp()

    single_worker_mock = unittest.mock.Mock()
    client = Client()
    parallel_runner = DaskParallelRunner(
        single_worker=single_worker_mock,
        dask_client=client,
        n_workers=1,
        output_directory=tmp_dir,
    )
    del parallel_runner

    assert not os.path.exists(os.path.join(tmp_dir, ".dask_scheduler_file"))
    assert client.status == "running"
    parallel_runner = DaskParallelRunner(
        single_worker=single_worker_mock,
        dask_client=client,
        n_workers=1,
        output_directory=tmp_dir,
    )
    del parallel_runner

    assert client.status == "running"
    client.shutdown()
