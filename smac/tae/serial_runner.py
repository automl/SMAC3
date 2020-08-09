import typing

from smac.configspace import Configuration
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType
from smac.tae.base import BaseRunner
from smac.utils.constants import MAXINT


class SerialRunner(BaseRunner):
    """Interface to submit and collect a job in a serial fashion.

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
        super(SerialRunner, self).__init__(
            ta=ta, stats=stats, run_obj=run_obj,
            par_factor=par_factor,
            cost_for_crash=cost_for_crash,
            abort_on_first_run_crash=abort_on_first_run_crash,
            n_workers=n_workers,
        )

        # No sense on having more than 1 worker
        assert self.n_workers == 1, "SerialRunner does not support more than 1 worker"

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
        try:
            self.results.put(
                self.run_wrapper(run_info)
            )
        except Exception as e:
            self.exceptions.append(e)

    def get_finished_runs(self) -> typing.List[typing.Tuple[RunInfo, RunValue]]:
        """This method returns any finished configuration, and returns a list with
        the results of exercising the configurations. This class keeps populating results
        to self.results until a call to get_finished runs is done. In this case, the
        self.results Queue is emptied and all RunValues produced by running self.run() are
        returned.

        Returns
        -------
            List[RunInfo, RunValue]: A list of RunInfo/RunValues pairs
            a submitted configuration
        """
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
        return

    def run(
        self, config: Configuration,
        instance: str,
        cutoff: typing.Optional[float] = None,
        seed: int = 12345,
        budget: typing.Optional[float] = None,
        instance_specific: str = "0",
    ) -> typing.Tuple[StatusType, float, float, typing.Dict]:
        """Runs target algorithm <self.ta> with configuration <config> on
        instance <instance> with instance specifics <specifics> for at most
        <cutoff> seconds and random seed <seed>

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
        return StatusType.SUCCESS, 12345.0, 1.2345, {}
