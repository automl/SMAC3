from abc import ABC, abstractmethod
import math
import time
import traceback
import typing

import numpy as np

from smac.configspace import Configuration
from smac.utils.constants import MAXINT
from smac.utils.logging import PickableLoggerAdapter
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class BaseRunner(ABC):
    """Interface class to handle the execution of SMAC' configurations.

    This interface defines how to interact with the SMBO loop.
    The complexity of running a configuration as well as handling the
    results is abstracted to the SMBO via a BaseRunner.

    From SMBO perspective, launching a configuration follows a
    submit/collect scheme as follows:

    1. A run is launched via submit_run()

       1. Submit_run internally calls run_wrapper(), a method that
          contains common processing functions among different runners,
          for example, handling capping and stats checking.

       2. A class that implements BaseRunner defines run() which is
          really the algorithm to translate a RunInfo to a RunValue, i.e.
          a configuration to an actual result.

    2. A completed run is collected via get_finished_runs(), which returns
       any finished runs, if any.

    3. This interface also offers the method wait() as a mechanism to make
       sure we have enough data in the next iteration to make a decision. For
       example, the intensifier might not be able to select the next challenger
       until more results are available.

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
        """
        Attributes
        ----------
        results
        ta
        stats
        run_obj
        par_factor
        cost_for_crash
        abort_first_run_crash

        Parameters
        ----------
        ta : typing.Union[typing.List[str], typing.Callable]
            target algorithm
        stats: Stats
             stats object to collect statistics about runtime/additional info
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

        # The results is a FIFO structure, implemented via a list
        # (because the Queue lock is not pickable). Finished runs are
        # put in this list and collected via process_finished_runs
        self.results = []  # type: typing.List[typing.Tuple[RunInfo, RunValue]]

        # Below state the support for a Runner algorithm that
        # implements a ta
        self.ta = ta
        self.stats = stats
        self.run_obj = run_obj
        self.par_factor = par_factor
        self.cost_for_crash = cost_for_crash
        self.abort_on_first_run_crash = abort_on_first_run_crash
        self.logger = PickableLoggerAdapter(
            self.__module__ + '.' + self.__class__.__name__)
        self._supports_memory_limit = False

        super().__init__()

    @abstractmethod
    def submit_run(self, run_info: RunInfo) -> None:
        """This function submits a configuration
        embedded in a RunInfo object, and uses one of the workers
        to produce a result (such result will eventually be available
        on the self.results FIFO).

        This interface method will be called by SMBO, with the expectation
        that a function will be executed by a worker.

        What will be executed is dictated by run_info, and "how" will it be
        executed is decided via the child class that implements a run() method.

        Because config submission can be a serial/parallel endeavor,
        it is expected to be implemented by a child class.

        Parameters
        ----------
        run_info: RunInfo
            An object containing the configuration and the necessary data to run it

        """
        pass

    @abstractmethod
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

    def run_wrapper(
        self,
        run_info: RunInfo,
    ) -> typing.Tuple[RunInfo, RunValue]:
        """Wrapper around run() to exec and check the execution of a given config file

        This function encapsulates common handling/processing, so that run() implementation
        is simplified.

        Parameters
        ----------
            run_info : RunInfo
                Object that contains enough information to execute a configuration run in
                isolation.

        Returns
        -------
            RunInfo:
                an object containing the configuration launched
            RunValue:
                Contains information about the status/performance of config
        """
        start = time.time()

        if run_info.cutoff is None and self.run_obj == "runtime":
            if self.logger:
                self.logger.critical(
                    "For scenarios optimizing running time "
                    "(run objective), a cutoff time is required, "
                    "but not given to this call."
                )
            raise ValueError(
                "For scenarios optimizing running time "
                "(run objective), a cutoff time is required, "
                "but not given to this call."
            )
        cutoff = None
        if run_info.cutoff is not None:
            cutoff = int(math.ceil(run_info.cutoff))

        try:
            status, cost, runtime, additional_info = self.run(
                config=run_info.config,
                instance=run_info.instance,
                cutoff=cutoff,
                seed=run_info.seed,
                budget=run_info.budget,
                instance_specific=run_info.instance_specific
            )
        except Exception as e:
            status = StatusType.CRASHED
            cost = self.cost_for_crash
            runtime = time.time() - start

            # Add context information to the error message
            exception_traceback = traceback.format_exc()
            error_message = repr(e)
            additional_info = {
                'traceback': exception_traceback,
                'error': error_message
            }

        end = time.time()

        if run_info.budget == 0 and status == StatusType.DONOTADVANCE:
            raise ValueError(
                "Cannot handle DONOTADVANCE state when using intensify or SH/HB on "
                "instances."
            )

        # Catch NaN or inf.
        if (
            self.run_obj == 'runtime' and not np.isfinite(runtime)
            or self.run_obj == 'quality' and not np.isfinite(cost)
        ):
            if self.logger:
                self.logger.warning("Target Algorithm returned NaN or inf as {}. "
                                    "Algorithm run is treated as CRASHED, cost "
                                    "is set to {} for quality scenarios. "
                                    "(Change value through \"cost_for_crash\""
                                    "-option.)".format(
                                        self.run_obj,
                                        self.cost_for_crash)
                                    )
            status = StatusType.CRASHED

        if self.run_obj == "runtime":
            # The following line pleases mypy - we already check for cutoff not being none above,
            # prior to calling run. However, mypy assumes that the data type of cutoff
            # is still Optional[int]
            assert cutoff is not None
            if runtime > self.par_factor * cutoff:
                self.logger.warning(
                    "Returned running time is larger "
                    "than {0} times the passed cutoff time. "
                    "Clamping to {0} x cutoff.".format(self.par_factor))
                runtime = cutoff * self.par_factor
                status = StatusType.TIMEOUT
            if status == StatusType.SUCCESS:
                cost = runtime
            else:
                cost = cutoff * self.par_factor
            if status == StatusType.TIMEOUT and run_info.capped:
                status = StatusType.CAPPED
        else:
            if status == StatusType.CRASHED:
                cost = self.cost_for_crash

        return run_info, RunValue(
            status=status,
            cost=cost,
            time=runtime,
            additional_info=additional_info,
            starttime=start,
            endtime=end
        )

    @abstractmethod
    def get_finished_runs(self) -> typing.List[typing.Tuple[RunInfo, RunValue]]:
        """This method returns any finished configuration, and returns a list with
        the results of exercising the configurations. This class keeps populating results
        to self.results until a call to get_finished runs is done. In this case, the
        self.results list is emptied and all RunValues produced by running run() are
        returned.

        Returns
        -------
            List[RunInfo, RunValue]: A list of pais RunInfo/RunValues
            a submitted configuration
        """
        raise NotImplementedError()

    @abstractmethod
    def wait(self) -> None:
        """SMBO/intensifier might need to wait for runs to finish before making a decision.
        This method waits until 1 run completes
        """
        pass

    @abstractmethod
    def pending_runs(self) -> bool:
        """
        Whether or not there are configs still running. Generally if the runner is serial,
        launching a run instantly returns it's result. On parallel runners, there might
        be pending configurations to complete.
        """
        pass

    @abstractmethod
    def num_workers(self) -> int:
        """
        Return the active number of workers that will execute tae runs.
        """
        pass
