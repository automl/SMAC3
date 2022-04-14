import typing

import os
import time
import warnings

import dask
from dask.distributed import Client, Future, wait

from smac.configspace import Configuration
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae import StatusType
from smac.tae.base import BaseRunner

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class DaskParallelRunner(BaseRunner):
    """Interface to submit and collect a job in a distributed fashion.

    DaskParallelRunner is intended to comply with the bridge design pattern.

    Nevertheless, to reduce the amount of code within single-vs-parallel
    implementations, DaskParallelRunner wraps a BaseRunner object which
    is then executed in parallel on n_workers.

    This class then is constructed by passing a BaseRunner that implements
    a run() method, and is capable of doing so in a serial fashion. Then,
    this wrapper class called DaskParallelRunner uses dask to initialize
    N number of BaseRunner that actively wait of a RunInfo to produce a
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
        DaskParallelRunner.get_finished_runs() and then passed to the SMBO.
    4.  Exceptions are also locally available to each worker and need to be
        collected.

    Dask works with Future object which are managed via the DaskParallelRunner.client.


    Parameters
    ----------
    single_worker: BaseRunner
        A runner to run in a distributed fashion
    n_workers: int
        Number of workers to use for distributed run. Will be ignored if ``dask_client`` is not ``None``.
    patience: int
        How much to wait for workers to be available if one fails
    output_directory: str, optional
        If given, this will be used for the dask worker directory and for storing server information.
        If a dask client is passed, it will only be used for storing server information as the
        worker directory must be set by the program/user starting the workers.
    dask_client: dask.distributed.Client
        User-created dask client, can be used to start a dask cluster and then attach SMAC to it.


    Attributes
    ----------
    results
    ta
    stats
    run_obj
    par_factor
    cost_for_crash
    abort_i_first_run_crash
    n_workers
    futures
    client
    """

    def __init__(
        self,
        single_worker: BaseRunner,
        n_workers: int,
        patience: int = 5,
        output_directory: typing.Optional[str] = None,
        dask_client: typing.Optional[dask.distributed.Client] = None,
    ):
        super(DaskParallelRunner, self).__init__(
            ta=single_worker.ta,
            stats=single_worker.stats,
            multi_objectives=single_worker.multi_objectives,
            run_obj=single_worker.run_obj,
            par_factor=single_worker.par_factor,
            cost_for_crash=single_worker.cost_for_crash,
            abort_on_first_run_crash=single_worker.abort_on_first_run_crash,
        )

        # The single worker, which is replicated on a need
        # basis to every compute node
        self.single_worker = single_worker
        self.n_workers = n_workers

        # How much time to wait for workers to be available
        self.patience = patience

        self.output_directory = output_directory

        # Because a run() method can have pynisher, we need to prevent the multiprocessing
        # workers to be instantiated as demonic - this cannot be passed via worker_kwargs
        dask.config.set({"distributed.worker.daemon": False})
        if dask_client is None:
            self.close_client_at_del = True
            self.client = Client(
                n_workers=self.n_workers,
                processes=True,
                threads_per_worker=1,
                local_directory=output_directory,
            )
            if self.output_directory:
                self.scheduler_file = os.path.join(self.output_directory, ".dask_scheduler_file")
                self.client.write_scheduler_file(scheduler_file=self.scheduler_file)
        else:
            self.close_client_at_del = False
            self.client = dask_client
        self.futures = []  # type: typing.List[Future]

        self.scheduler_info = self.client._get_scheduler_info()

    def submit_run(self, run_info: RunInfo) -> None:
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
        if not self._workers_available():
            futures = wait(self.futures, return_when="FIRST_COMPLETED")
            futures.done
            self._extract_completed_runs_from_futures()

        # In code check to make sure that there are resources
        if not self._workers_available():
            warnings.warn("No workers are available. This could mean workers crashed" "Waiting for new workers...")
            time.sleep(self.patience)
            if not self._workers_available():
                raise ValueError(
                    "Tried to execute a job, but no worker was "
                    "available. This likely means that a worker crashed "
                    "or no workers were properly configured."
                )

        # At this point we can submit the job
        # For `pure=False`, see
        #   http://distributed.dask.org/en/stable/client.html#pure-functions-by-default
        self.futures.append(self.client.submit(self.single_worker.run_wrapper, run_info, pure=False))

    def get_finished_runs(self) -> typing.List[typing.Tuple[RunInfo, RunValue]]:
        """This method returns any finished configuration, and returns a list with the results of
        exercising the configurations. This class keeps populating results to self.results until a
        call to get_finished runs is done. In this case, the self.results list is emptied and all
        RunValues produced by running run() are returned.

        Returns
        -------
            List[RunInfo, RunValue]: A list of RunValues (and respective RunInfo), that is,
                the results of executing a run_info
            a submitted configuration
        """
        # Proactively see if more configs have finished
        self._extract_completed_runs_from_futures()

        results_list = []
        while self.results:
            results_list.append(self.results.pop())
        return results_list

    def _extract_completed_runs_from_futures(self) -> None:
        """A run is over, when a future has done() equal true. This function collects the completed
        futures and move them from self.futures to self.results.

        We make sure futures never exceed the capacity of the scheduler
        """
        # In code check to make sure we don;t exceed resource allocation
        if len(self.futures) > sum(self.client.nthreads().values()):
            warnings.warn(
                "More running jobs than resources available "
                "Should not have more futures/runs in remote workers "
                "than the number of workers. This could mean a worker "
                "crashed and was not able to be recovered by dask. "
            )

        # A future is removed to the list of futures as an indication
        # that a worker is available to take in an extra job
        done_futures = [f for f in self.futures if f.done()]
        for future in done_futures:
            self.results.append(future.result())
            self.futures.remove(future)

    def wait(self) -> None:
        """SMBO/intensifier might need to wait for runs to finish before making a decision.

        This class waits until 1 run completes
        """
        if self.futures:
            futures = wait(self.futures, return_when="FIRST_COMPLETED")
            futures.done

    def pending_runs(self) -> bool:
        """Whether or not there are configs still running.

        Generally if the runner is serial, launching a run instantly returns it's result. On
        parallel runners, there might be pending configurations to complete.
        """
        # If there are futures available, it translates
        # to runs still not finished/processed
        return len(self.futures) > 0

    def run(
        self,
        config: Configuration,
        instance: str,
        cutoff: typing.Optional[float] = None,
        seed: int = 12345,
        budget: typing.Optional[float] = None,
        instance_specific: str = "0",
    ) -> typing.Tuple[StatusType, float, float, typing.Dict]:
        """This method only complies with the abstract parent class. In the parallel case, we call
        the single worker run() method.

        Parameters
        ----------
            config : Configuration
                dictionary param -> value
            instance : string
                problem instance
            cutoff : float, optional
                Wallclock time limit of the target algorithm. If no value is
                provided no limit will be enforced.
            seed : int
                random seed
            budget : float, optional
                A positive, real-valued number representing an arbitrary limit to the target
                algorithm. Handled by the target algorithm internally
            instance_specific: str
                instance specific information (e.g., domain file or solution)

        Returns
        -------
            status: enum of StatusType (int)
                {SUCCESS, TIMEOUT, CRASHED, ABORT}
            cost: float
                cost/regret/quality (float) (None, if not returned by TA)
            runtime: float
                runtime (None if not returned by TA)
            additional_info: dict
                all further additional run information
        """
        return self.single_worker.run(
            config=config,
            instance=instance,
            cutoff=cutoff,
            seed=seed,
            budget=budget,
            instance_specific=instance_specific,
        )

    def num_workers(self) -> int:
        """Total number of workers available.

        This number is dynamic as more resources can be allocated
        """
        return sum(self.client.nthreads().values())

    def _workers_available(self) -> bool:
        """
        Query if there are workers available, which means that there are resources to launch a
        dask job.
        """
        total_compute_power = sum(self.client.nthreads().values())
        if len(self.futures) < total_compute_power:
            return True
        return False

    def __del__(self) -> None:
        """
        Make sure that when this object gets deleted, the client is terminated.

        This is only done if the client was created by the dask runner.
        """
        if self.close_client_at_del:
            self.client.close()
