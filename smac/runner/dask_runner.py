from __future__ import annotations

from typing import Any, Iterator

import time
from pathlib import Path

import dask
from ConfigSpace import Configuration
from dask.distributed import Client, Future, wait

from smac.runhistory import StatusType, TrialInfo, TrialValue
from smac.runner.abstract_runner import AbstractRunner
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class DaskParallelRunner(AbstractRunner):
    """Interface to submit and collect a job in a distributed fashion. DaskParallelRunner is
    intended to comply with the bridge design pattern. Nevertheless, to reduce the amount of code
    within single-vs-parallel implementations, DaskParallelRunner wraps a BaseRunner object which
    is then executed in parallel on `n_workers`.

    This class then is constructed by passing an AbstractRunner that implements
    a `run` method, and is capable of doing so in a serial fashion. Next,
    this wrapper class uses dask to initialize `N` number of AbstractRunner that actively wait of a
    TrialInfo to produce a RunInfo object.

    To be more precise, the work model is then:

    1. The intensifier dictates "what" to run (a configuration/instance/seed) via a TrialInfo object.
    2. An abstract runner takes this TrialInfo object and launches the task via
       `submit_run`. In the case of DaskParallelRunner, `n_workers` receive a pickle-object of
       `DaskParallelRunner.single_worker`, each with a `run` method coming from
       `DaskParallelRunner.single_worker.run()`
    3. TrialInfo objects are run in a distributed fashion, and their results are available locally to each worker. The
       result is collected by `iter_results` and then passed to SMBO.
    4. Exceptions are also locally available to each worker and need to be collected.

    Dask works with `Future` object which are managed via the DaskParallelRunner.client.

    Parameters
    ----------
    single_worker : AbstractRunner
        A runner to run in a distributed fashion. Will be distributed using `n_workers`.
    patience: int, default to 5
        How much to wait for workers (seconds) to be available if one fails.
    dask_client: Client | None, defaults to None
        User-created dask client, which can be used to start a dask cluster and then attach SMAC to it. This will not
        be closed automatically and will have to be closed manually if provided explicitly. If none is provided
        (default), a local one will be created for you and closed upon completion.
    """

    def __init__(
        self,
        single_worker: AbstractRunner,
        patience: int = 5,
        dask_client: Client | None = None,
    ):
        super().__init__(
            scenario=single_worker._scenario,
            required_arguments=single_worker._required_arguments,
        )

        # The single worker to hold on to and call run on
        self._single_worker = single_worker

        # The list of futures that dask will use to indicate in progress runs
        self._pending_trials: list[Future] = []

        # Dask related variables
        self._scheduler_file: Path | None = None
        self._patience = patience

        self._client: Client
        self._close_client_at_del: bool

        if dask_client is None:
            dask.config.set({"distributed.worker.daemon": False})
            self._close_client_at_del = True
            self._client = Client(
                n_workers=self._scenario.n_workers,
                processes=True,
                threads_per_worker=1,
                local_directory=str(self._scenario.output_directory),
            )

            if self._scenario.output_directory is not None:
                self._scheduler_file = Path(self._scenario.output_directory, ".dask_scheduler_file")
                self._client.write_scheduler_file(scheduler_file=str(self._scheduler_file))
        else:
            # We just use their set up
            self._client = dask_client
            self._close_client_at_del = False

    def submit_trial(self, trial_info: TrialInfo, **dask_data_to_scatter: dict[str, Any]) -> None:
        """This function submits a configuration embedded in a ``trial_info`` object, and uses one of
        the workers to produce a result locally to each worker.

        The execution of a configuration follows this procedure:

        #. The SMBO/intensifier generates a `TrialInfo`.
        #. SMBO calls `submit_trial` so that a worker launches the `trial_info`.
        #. `submit_trial` internally calls ``self.run()``. It does so via a call to `run_wrapper` which contains common
           code that any `run` method will otherwise have to implement.

        All results will be only available locally to each worker, so the main node needs to collect them.

        Parameters
        ----------
        trial_info : TrialInfo
            An object containing the configuration launched.

        dask_data_to_scatter: dict[str, Any]
            When a user scatters data from their local process to the distributed network,
            this data is distributed in a round-robin fashion grouping by number of cores.
            Roughly speaking, we can keep this data in memory and then we do not have to (de-)serialize the data
            every time we would like to execute a target function with a big dataset.
            For example, when your target function has a big dataset shared across all the target function,
            this argument is very useful.
        """
        # Check for resources or block till one is available
        if self.count_available_workers() <= 0:
            logger.debug("No worker available. Waiting for one to be available...")
            wait(self._pending_trials, return_when="FIRST_COMPLETED")
            self._process_pending_trials()

        # Check again to make sure that there are resources
        if self.count_available_workers() <= 0:
            logger.warning("No workers are available. This could mean workers crashed. Waiting for new workers...")
            time.sleep(self._patience)
            if self.count_available_workers() <= 0:
                raise RuntimeError(
                    "Tried to execute a job, but no worker was ever available."
                    "This likely means that a worker crashed or no workers were properly configured."
                )

        # At this point we can submit the job
        trial = self._client.submit(self._single_worker.run_wrapper, trial_info=trial_info, **dask_data_to_scatter)
        self._pending_trials.append(trial)

    def iter_results(self) -> Iterator[tuple[TrialInfo, TrialValue]]:  # noqa: D102
        self._process_pending_trials()
        while self._results_queue:
            yield self._results_queue.pop(0)

    def wait(self) -> None:  # noqa: D102
        if self.is_running():
            wait(self._pending_trials, return_when="FIRST_COMPLETED")

    def is_running(self) -> bool:  # noqa: D102
        return len(self._pending_trials) > 0

    def run(
        self,
        config: Configuration,
        instance: str | None = None,
        budget: float | None = None,
        seed: int | None = None,
        **dask_data_to_scatter: dict[str, Any],
    ) -> tuple[StatusType, float | list[float], float, dict]:  # noqa: D102
        return self._single_worker.run(
            config=config, instance=instance, seed=seed, budget=budget, **dask_data_to_scatter
        )

    def count_available_workers(self) -> int:
        """Total number of workers available. This number is dynamic as more resources
        can be allocated.
        """
        return sum(self._client.nthreads().values()) - len(self._pending_trials)

    def close(self, force: bool = False) -> None:
        """Closes the client."""
        if self._close_client_at_del or force:
            self._client.close()

    def _process_pending_trials(self) -> None:
        """The completed trials are moved from ``self._pending_trials`` to ``self._results_queue``.
        We make sure pending trials never exceed the capacity of the scheduler.
        """
        # In code check to make sure we don't exceed resource allocation
        if self.count_available_workers() < 0:
            logger.warning(
                "More running jobs than resources available. "
                "Should not have more pending trials in remote workers "
                "than the number of workers. This could mean a worker "
                "crashed and was not able to be recovered by dask. "
            )

        # Move the done run from the worker to the results queue
        done = [trial for trial in self._pending_trials if trial.done()]
        for trial in done:
            self._results_queue.append(trial.result())
            self._pending_trials.remove(trial)

    def __del__(self) -> None:
        """Makes sure that when this object gets deleted, the client is terminated. This
        is only done if the client was created by the dask runner.
        """
        if self._close_client_at_del:
            self.close()
