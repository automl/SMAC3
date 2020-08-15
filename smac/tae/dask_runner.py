import typing

import dask
from dask.distributed import Client, Future, wait

from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae.base import BaseRunner


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
    1- The smbo.intensifier dictates "what" to run (a configuration/instance/seed)
       via a RunInfo object.
    2- a tae_runner takes this RunInfo object and launches the task via
       tae_runner.submit_run(). In the case of DaskParallelRunner, n_workers
       receive a pickle-object of DaskParallelRunner.single_worker, each with a
       run() method coming from DaskParallelRunner.single_worker.run()
    3- RunInfo objects are run in a distributed fashion, an their results are
       available locally to each worker. Such result is collected by
       DaskParallelRunner.get_finished_runs() and then passed to the SMBO.
    4- Exceptions are also locally available to each worker and need to be
       collected.

    Dask works with Future object which are managed via the DaskParallelRunner.client.


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

    Parameters
    ---------
    ta : list
        target algorithm command line as list of arguments
    stats: Stats()
         stats object to collect statistics about runtime and so on
    run_obj: str
        run objective of SMAC
    par_factor: int
        penalization factor
    cost_for_crash : float
        cost that is used in case of crashed runs (including runs
        that returned NaN or inf)
    abort_on_first_run_crash: bool
        if true and first run crashes, raise FirstRunCrashedException
    n_workers: int
        Number of workers that will actively be running configurations in parallel
    """
    def __init__(
        self,
        single_worker: BaseRunner,
        n_workers: int = 1
    ):
        super(DaskParallelRunner, self).__init__(
            ta=single_worker.ta,
            stats=single_worker.stats,
            run_obj=single_worker.run_obj,
            par_factor=single_worker.par_factor,
            cost_for_crash=single_worker.cost_for_crash,
            abort_on_first_run_crash=single_worker.abort_on_first_run_crash,
        )

        # The single worker, which is replicated on a need
        # basis to every compute node
        self.single_worker = single_worker

        # n_workers defines the number of workers that a client is originally initialized
        # with. This number is dynamic and can vary by adding more workers to the proper
        # scheduler address.
        # Because a run() method can have pynisher, we need to prevent the multiprocessing
        # workers to be instantiated as demonic
        self.n_workers = n_workers
        dask.config.set({'distributed.worker.daemon': False})
        self.client = Client(n_workers=self.n_workers, processes=True, threads_per_worker=1)
        self.futures = []  # type: typing.List[Future]

    def submit_run(self, run_info: RunInfo) -> None:
        """This function submits a configuration
        embedded in a run_info object, and uses one of the workers
        to produce a result locally to each worker.

        The execution of a configuration follows this procedure:
        1- SMBO/intensifier generates a run_info
        2- SMBO calls submit_run so that a worker launches the run_info
        3- submit_run internally calls self.run(). it does so via a call to self.run_wrapper()
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
            done_futures = wait(self.futures, return_when='FIRST_COMPLETED').done
            for future in done_futures:
                self.results.append(future.result())

        # At this point we can submit the job
        self.futures.append(
            self.client.submit(
                self.single_worker.run_wrapper,
                run_info
            )
        )

    def get_finished_runs(self) -> typing.List[typing.Tuple[RunInfo, RunValue]]:
        """This method returns any finished configuration, and returns a list with
        the results of exercising the configurations. This class keeps populating results
        to self.results until a call to get_finished runs is done. In this case, the
        self.results list is emptied and all RunValues produced by running run() are
        returned.

        Returns
        -------
            List[RunInfo, RunValue]: A list of RunValues (and respective RunInfo), that is,
                the results of executing a run_info
            a submitted configuration
        """

        done_futures = [f for f in self.futures if f.done()]
        for future in done_futures:
            self.results.append(future.result())

            self.futures.remove(future)

        results_list = []
        while self.results:
            results_list.append(self.results.pop())
        return results_list

    def wait(self) -> None:
        """SMBO/intensifier might need to wait for runs to finish before making a decision.
        This class waits until 1 run completes
        """
        if self.futures:
            wait(self.futures, return_when='FIRST_COMPLETED').done

    def _workers_available(self) -> bool:
        """"Query if there are workers available, which means
        that there are resources to launch a dask job"""
        total_compute_power = sum(self.client.nthreads().values())
        if len(self.futures) < total_compute_power:
            return True
        return False

    def __del__(self) -> None:
        """Make sure that when this object gets deleted, the client is terminated."""
        self.client.close()
