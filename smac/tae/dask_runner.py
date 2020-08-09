import time
import typing

import dask
from dask.distributed import Client, Future, wait

from smac.configspace import Configuration
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType
from smac.tae.base import BaseRunner
from smac.utils.constants import MAXINT


class DaskParallelRunner(BaseRunner):
    """Interface to submit and collect a job in a distributed fashion.

    Attributes
    ----------

    results: Queue
        A worker store it's result in a queue. SMBO collects the results of a completed
        operation from the queue.
    exceptions:
        A list of capturing exceptions, product of the failure of submit/exec a job
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
        ta: typing.Union[typing.List[str], typing.Callable],
        stats: Stats,
        run_obj: str = "runtime",
        par_factor: int = 1,
        cost_for_crash: float = float(MAXINT),
        abort_on_first_run_crash: bool = True,
        n_workers: int = 1
    ):
        super(DaskParallelRunner, self).__init__(
            ta=ta, stats=stats, run_obj=run_obj, par_factor=par_factor,
            cost_for_crash=cost_for_crash,
            abort_on_first_run_crash=abort_on_first_run_crash,
            n_workers=n_workers
        )

        dask.config.set({'distributed.worker.daemon': False})
        self.client = Client(n_workers=self.n_workers, processes=True, threads_per_worker=1)
        self.futures = []  # type: typing.List[Future]
        self.counter = 0

    def submit_run(self, run_info: RunInfo) -> None:
        """This function submits a configuration
        embedded in a run_info object, and uses one of the workers
        to produce a result that is stored in a results queue.

        self.func is not yet a completely isolated function that can
        take a run_info and produce a result. The current SMAC infrastructure is
        created based on an object called tae that has the necessary attributes to
        execute a run_info via it's run() method.

        In other words, the execution of a configuration follows this procedure:
        1- SMBO/intensifier generates a run_info
        2- SMBO calls submit_run so that a worker launches the run_info
        3- submit_run internally calls self.run(). it does so via a call to self.run_wrapper()
        which contains common code that any run() method will otherwise have to implement, like
        capping check.

        Child classes must implement a run() method.

        Parameters
        ----------
        run_info: RunInfo
            An object containing the configuration and the necessary data to run it

        """
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        # Check for resources or block till one is available
        print("\n\n\n")
        if not self._workers_available():
            print("Force collect a future {now}")
            done_futures = wait(self.futures, return_when='FIRST_COMPLETED').done
            for future in done_futures:
                if future.exception():
                    self.exceptions.append(future.exception)
                self.results.put(future.result())

        # At this point we can submit the job
        print(f"Scheduling job {self.counter} at {now}")
        self.counter += 1
        self.futures.append(
            self.client.submit(
                self.run_wrapper,
                run_info
            )
        )

    def get_finished_runs(self) -> typing.List[typing.Tuple[RunInfo, RunValue]]:
        """This method returns any finished configuration, and returns a list with
        the results of exercising the configurations. This class keeps populating results
        to self.results until a call to get_finished runs is done. In this case, the
        self.results Queue is emptied and all RunValues produced by running self.func are
        returned.

        Returns
        -------
            List[RunInfo, RunValue]: A list of RunValues (and respective RunInfo), that is,
                the results of executing a run_info
            a submitted configuration
        """

        done_futures = [f for f in self.futures if f.done()]
        for future in done_futures:
            if future.exception():
                self.exceptions.append(future.exception)
            self.results.put(future.result())
            self.futures.remove(future)

        results_list = []
        while not self.results.empty():
            results_list.append(self.results.get())
        return results_list

    def get_exceptions(self) -> typing.List[Exception]:
        """In case of parallel execution, the exceptions need to be collected.
        When a job finished, exceptions (if any) are added to a list.
        This method returns such list.

        Returns:
        -------
            List[Exception]: List of all exceptions that occurred when launching a job.
        """
        return self.exceptions

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
            print(f"There are workers {total_compute_power} and {len(self.futures)}")
            return True
        print(f"There are NO workers {total_compute_power}")
        return False

    def __del__(self) -> None:
        """Make sure that when this object gets deleted, the client is terminated."""
        self.client.close()
