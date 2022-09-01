from __future__ import annotations

from typing import Any, Iterator

import time
from pathlib import Path

import dask
from dask.distributed import Client, Future, wait

from smac.configspace import Configuration
from smac.runhistory import TrialInfo, TrialValue, StatusType
from smac.runner.abstract_runner import AbstractRunner
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class DaskParallelRunner(AbstractRunner):
    """Interface to submit and collect a job in a distributed fashion.

    DaskParallelRunner is intended to comply with the bridge design pattern.

    Nevertheless, to reduce the amount of code within single-vs-parallel
    implementations, DaskParallelRunner wraps a BaseRunner object which
    is then executed in parallel on n_workers.

    This class then is constructed by passing a AbstractRunner that implements
    a run() method, and is capable of doing so in a serial fashion. Then,
    this wrapper class called DaskParallelRunner uses dask to initialize
    N number of AbstractRunner that actively wait of a RunInfo to produce a
    RunValue object.

    To be more precise, the work model is then:
    1.  The smbo.intensifier dictates "what" to run (a configuration/instance/seed)
        via a RunInfo object.
    2.  a tae_runner takes this RunInfo object and launches the task via
        tae_runner.submit_run(). In the case of DaskParallelRunner, n_workers
        receive a pickle-object of DaskParallelRunner.single_worker, each with a
        run() method coming from DaskParallelRunner.single_worker.run()
    3.  RunInfo objects are run in a distributed fashion, an their results are
        available locally to each worker. Such result is collected by
        DaskParallelRunner.iter_results() and then passed to the SMBO.
    4.  Exceptions are also locally available to each worker and need to be
        collected.

    Dask works with Future object which are managed via the DaskParallelRunner.client.

    Parameters
    ----------
    single_worker: AbstractRunner
        A runner to run in a distributed fashion, will be distributed using ``n_workers``
    n_workers: int
        Number of workers to use for distributed run.
        Will be ignored if ``dask_client`` is not ``None``.
    patience: int, default to 5
        How much to wait for workers (seconds) to be available if one fails
    output_directory: str | Path, optional
        If given, this will be used for the dask worker directory and for storing server
        information. If a dask client is passed, it will only be used for storing server
        information as the worker directory must be set by the program/user starting the
        workers.
    dask_client: Client | None, optional
        User-created dask client, can be used to start a dask cluster and then attach
        SMAC to it. This will not be closed automatically and will have to be closed
        manually if provided explicitly.
        If None is provided (default), a local one will be created for you and closed
        upon completion.
    """

    def __init__(
        self,
        single_worker: AbstractRunner,
        patience: int = 5,
        dask_client: Client | None = None,
    ):
        super().__init__(
            target_algorithm=single_worker.target_algorithm,
            scenario=single_worker.scenario,
            stats=single_worker.stats,
        )

        # The single worker to hold on to and call run on
        self._single_worker = single_worker

        # The list of futures that dask will use to indicate in progress runs
        self._pending_runs: list[Future] = []

        # Dask related variables
        self._scheduler_file: Path | None = None
        self._patience = patience

        self._client: Client
        self._close_client_at_del: bool

        if dask_client is None:
            dask.config.set({"distributed.worker.daemon": False})
            self._close_client_at_del = True
            self._client = Client(
                n_workers=self.scenario.n_workers,
                processes=True,
                threads_per_worker=1,
                local_directory=str(self.scenario.output_directory),
            )
            if self.scenario.output_directory is not None:
                self._scheduler_file = self.scenario.output_directory / ".dask_scheduler_file"
                self._client.write_scheduler_file(scheduler_file=str(self._scheduler_file))
        else:
            # We just use their set up
            self._client = dask_client
            self._close_client_at_del = False

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "code": self.target_algorithm.__code__.co_code,
        }

    def submit_run(self, run_info: TrialInfo) -> None:
        """This function submits a configuration embedded in a run_info object, and uses one of the
        workers to produce a result locally to each worker.

        The execution of a configuration follows this procedure:
        1.  SMBO/intensifier generates a run_info
        2.  SMBO calls submit_run so that a worker launches the run_info
        3.  submit_run internally calls self.run(). it does so via a call to self.run_wrapper()
        which contains common code that any run() method will otherwise have to implement, like
        capping check.

        Child classes must implement a run() method.
        All results will be only available locally to each worker, so the
        main node needs to collect them.

        Parameters
        ----------
        run_info: RunInfo
            An object containing the configuration and the necessary data to run it
        """
        # Check for resources or block till one is available
        if self.available_worker_count() <= 0:
            wait(self._pending_runs, return_when="FIRST_COMPLETED")
            self._process_pending_runs()

        # Check again to make sure that there are resources
        if self.available_worker_count() <= 0:
            logger.warning("No workers are available. This could mean workers crashed. Waiting for new workers...")
            time.sleep(self._patience)
            if self.available_worker_count() <= 0:
                raise RuntimeError(
                    "Tried to execute a job, but no worker was ever available."
                    "This likely means that a worker crashed or no workers were properly configured."
                )

        # At this point we can submit the job
        run = self._client.submit(self._single_worker.run_wrapper, run_info=run_info)
        self._pending_runs.append(run)

    def iter_results(self) -> Iterator[tuple[TrialInfo, TrialValue]]:
        """This method returns any finished configuration, and returns a list with the results of
        exercising the configurations. This class keeps populating results to self._reseults_queue until a
        call to get_finished runs is done. In this case, the self._reseults_queue list is emptied and all
        RunValues produced by running run() are returned.

        Returns
        -------
        Iterator[tuple[RunInfo, RunValue]]
            A list of RunValues and RunInfo, which are the results of executing the submitted configurations.
        """
        self._process_pending_runs()
        while self._results_queue:
            yield self._results_queue.pop(0)

    def wait(self) -> None:
        """SMBO/intensifier might need to wait for runs to finish before making a decision.

        This class waits until 1 run completes
        """
        if self.is_running():
            wait(self._pending_runs, return_when="FIRST_COMPLETED")

    def is_running(self) -> bool:
        """Whether this runner is currently running something"""
        return len(self._pending_runs) > 0

    def run(
        self,
        config: Configuration,
        instance: str | None = None,
        seed: int = 0,
        budget: float | None = None,
    ) -> tuple[StatusType, float | list[float], float, dict]:
        """This method only complies with the abstract parent class. In the parallel
        case, we call the single worker run() method.
        """
        return self._single_worker.run(config=config, instance=instance, seed=seed, budget=budget)

    def available_worker_count(self) -> int:
        """Total number of workers available. This number is dynamic as more resources can be allocated."""
        return sum(self._client.nthreads().values()) - len(self._pending_runs)

    def close(self, force: bool = False) -> None:
        """Close the associated client"""
        if self._close_client_at_del or force:
            self._client.close()

    def _process_pending_runs(self) -> None:
        """A run is over, indiciated by done() equal true. This function collects
        the completed runs and move them from self._pending_runs to self._reseults_queue.

        We make sure pending runs never exceed the capacity of the scheduler
        """
        # In code check to make sure we don;t exceed resource allocation
        if self.available_worker_count() < 0:
            logger.warning(
                "More running jobs than resources available. "
                "Should not have more pending runs in remote workers "
                "than the number of workers. This could mean a worker "
                "crashed and was not able to be recovered by dask. "
            )

        # Move the done run from the worker to the results queue
        done = [run for run in self._pending_runs if run.done()]
        for run in done:
            self._results_queue.append(run.result())
            self._pending_runs.remove(run)

    def __del__(self) -> None:
        """Make sure that when this object gets deleted, the client is terminated. This
        is only done if the client was created by the dask runner.
        """
        if self._close_client_at_del:
            self.close()
