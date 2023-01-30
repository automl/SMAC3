from __future__ import annotations

from typing import Iterator

from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runner.abstract_runner import AbstractRunner


class AbstractSerialRunner(AbstractRunner):
    def submit_trial(self, trial_info: TrialInfo) -> None:
        """This function submits a trial_info object in a serial fashion. As there is a single
         worker for this task, this interface can be considered a wrapper over the `run` method.

        Both result/exceptions can be completely determined in this step so both lists
        are properly filled.

        Parameters
        ----------
        trial_info : TrialInfo
            An object containing the configuration launched.
        """
        self._results_queue.append(self.run_wrapper(trial_info))

    def iter_results(self) -> Iterator[tuple[TrialInfo, TrialValue]]:  # noqa: D102
        while self._results_queue:
            yield self._results_queue.pop(0)

    def wait(self) -> None:
        """The SMBO/intensifier might need to wait for trials to finish before making a decision.
        For serial runners, no wait is needed as the result is immediately available.
        """
        # There is no need to wait in serial runners. When launching a trial via submit, as
        # the serial trial uses the same process to run, the result is always available
        # immediately after. This method implements is just an implementation of the
        # abstract method via a simple return, again, because there is no need to wait
        return

    def is_running(self) -> bool:  # noqa: D102
        return False

    def count_available_workers(self) -> int:
        """Returns the number of available workers. Serial workers only have one worker."""
        return 1
