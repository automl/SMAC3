from __future__ import annotations

from typing import Iterator

from smac.runhistory import RunInfo, RunValue
from smac.runner.runner import AbstractRunner

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class SerialRunner(AbstractRunner):
    """Interface to submit and collect a job in a serial fashion.

    Dictates what a worker should do to convert a configuration/instance/seed to a result.

    This class is expected to be extended via the implementation of a run() method for
    the desired task.
    """

    def submit_run(self, run_info: RunInfo) -> None:
        """This function submits a run_info object in a serial fashion.

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
        self._results_queue.append(self.run_wrapper(run_info))

    def iter_results(self) -> Iterator[tuple[RunInfo, RunValue]]:
        """This method returns any finished configuration, and returns a list with the
        results of exercising the configurations. This class keeps populating results to
        self._results_queue until a call to get_finished runs is done. In this case,
        the self._results_queue list is emptied and all RunValues produced by running
        self.run() are returned.

        Returns
        -------
        list[tuple[RunInfo, RunValue]]
            A list of RunInfo/RunValues pairs a submitted configuration.
        """
        while self._results_queue:
            yield self._results_queue.pop()

    def wait(self) -> None:
        """
        SMBO/intensifier might need to wait for runs to finish before making a decision.
        For serial runs, no wait is needed as the result is immediately available.
        """
        # There is no need to wait in serial runs. When launching a run via submit, as
        # the serial run uses the same process to run, the result is always available
        # immediately after. This method implements is just an implementation of the
        # abstract method via a simple return, again, because there is no need to wait
        return

    def is_running(self) -> bool:
        """Whether or not there are configs still running.

        Generally if the runner is serial, launching a run instantly returns it's result.
        On parallel runners, there might be pending configurations to complete.
        """
        return False

    def available_worker_count(self) -> int:
        """Total number of workers available. Serial workers only have 1"""
        return 1
