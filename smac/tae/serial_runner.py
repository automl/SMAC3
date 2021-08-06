import typing

from smac.configspace import Configuration
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType
from smac.tae.base import BaseRunner
from smac.utils.constants import MAXINT

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class SerialRunner(BaseRunner):
    """Interface to submit and collect a job in a serial fashion.


    It dictates what a worker should do to convert a
    configuration/instance/seed to a result.

    This class is expected to be extended via the implementation of
    a run() method for the desired task.

    Attributes
    ----------

    results
    ta
    stats
    run_obj
    par_factor
    cost_for_crash
    abort_i_first_run_crash

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
    """

    def __init__(
        self,
        ta: typing.Union[typing.List[str], typing.Callable],
        stats: Stats,
        run_obj: str = "runtime",
        par_factor: int = 1,
        cost_for_crash: float = float(MAXINT),
        abort_on_first_run_crash: bool = True,
    ):
        super(SerialRunner, self).__init__(
            ta=ta, stats=stats, run_obj=run_obj,
            par_factor=par_factor,
            cost_for_crash=cost_for_crash,
            abort_on_first_run_crash=abort_on_first_run_crash,
        )

    def submit_run(self, run_info: RunInfo) -> None:
        """This function submits a run_info object
        in a serial fashion.

        As there is a single worker for this task, this
        interface can be considered a wrapper over the run()
        method.

        Both result/exceptions can be completely determined in this
        step so both lists are properly filled.

        Parameters
        ----------
        run_info: RunInfo
            An object containing the configuration and the necessary data to run it

        """
        self.results.append(
            self.run_wrapper(run_info)
        )

    def get_finished_runs(self) -> typing.List[typing.Tuple[RunInfo, RunValue]]:
        """This method returns any finished configuration, and returns a list with
        the results of exercising the configurations. This class keeps populating results
        to self.results until a call to get_finished runs is done. In this case, the
        self.results list is emptied and all RunValues produced by running self.run() are
        returned.

        Returns
        -------
            List[RunInfo, RunValue]: A list of RunInfo/RunValues pairs
            a submitted configuration
        """
        results_list = []
        while self.results:
            results_list.append(self.results.pop())
        return results_list

    def wait(self) -> None:
        """SMBO/intensifier might need to wait for runs to finish before making a decision.
        For serial runs, no wait is needed as the result is immediately available.
        """

        # There is no need to wait in serial runs.
        # When launching a run via submit, as the serial run
        # uses the same process to run, the result is always available
        # immediately after. This method implements is just an implementation of the
        # abstract method via a simple return, again, because there is
        # no need to wait (as in distributed runs)
        return

    def pending_runs(self) -> bool:
        """
        Whether or not there are configs still running. Generally if the runner is serial,
        launching a run instantly returns it's result. On parallel runners, there might
        be pending configurations to complete.
        """
        # No pending runs in a serial run. Execution is blocking
        return False

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

        This method exemplifies how to defined the run() method

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
        pass

    def num_workers(self) -> int:
        """Total number of workers available."""
        # Any serial runner supports only 1 worker
        return 1
