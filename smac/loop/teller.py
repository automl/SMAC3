import numpy as np
from smac.runhistory import RunInfo, RunValue, StatusType
from smac.utils.logging import get_logger
from smac.runner.exceptions import (
    FirstRunCrashedException,
    TargetAlgorithmAbortException,
)


logger = get_logger(__name__)


class Teller:
    def tell(self, run_info: RunInfo, run_value: RunValue, time_left: float, save: bool = True) -> None:
        # We removed `abort_on_first_run_crash` and therefore we expect the first
        # run to always succeed.
        if self.stats.finished == 0 and run_value.status == StatusType.CRASHED:
            additional_info = ""
            if "traceback" in run_value.additional_info:
                additional_info = "\n\n" + run_value.additional_info["traceback"]

            raise FirstRunCrashedException("The first run crashed. Please check your setup again." + additional_info)

        # Update SMAC stats
        self.stats.target_algorithm_walltime_used += float(run_value.time)
        self.stats.finished += 1

        logger.debug(
            f"Status: {run_value.status}, cost: {run_value.cost}, time: {run_value.time}, "
            f"Additional: {run_value.additional_info}"
        )

        self.runhistory.add(
            config=run_info.config,
            cost=run_value.cost,
            time=run_value.time,
            status=run_value.status,
            instance=run_info.instance,
            seed=run_info.seed,
            budget=run_info.budget,
            starttime=run_value.starttime,
            endtime=run_value.endtime,
            force_update=True,
            additional_info=run_value.additional_info,
        )
        self.stats.n_configs = len(self.runhistory.config_ids)

        if run_value.status == StatusType.ABORT:
            raise TargetAlgorithmAbortException(
                "The target algorithm was aborted. The last incumbent can be found in the trajectory file."
            )
        elif run_value.status == StatusType.STOP:
            self._stop = True
            return

        # Update the intensifier with the result of the runs
        self.incumbent, _ = self.intensifier.process_results(
            run_info=run_info,
            run_value=run_value,
            incumbent=self.incumbent,
            runhistory=self.runhistory,
            time_bound=max(self._min_time, time_left),
        )

        # Gracefully end optimization if termination cost is reached
        if self.scenario.termination_cost_threshold != np.inf:
            if not isinstance(run_value.cost, list):
                cost = [run_value.cost]
            else:
                cost = run_value.cost

            if not isinstance(self.scenario.termination_cost_threshold, list):
                cost_threshold = [self.scenario.termination_cost_threshold]
            else:
                cost_threshold = self.scenario.termination_cost_threshold

            if len(cost) != len(cost_threshold):
                raise RuntimeError("You must specify a termination cost threshold for each objective.")

            if all(cost[i] < cost_threshold[i] for i in range(len(cost))):
                self._stop = True

        for callback in self._callbacks:
            response = callback.on_iteration_end(smbo=self, info=run_info, value=run_value)

            # If a callback returns False, the optimization loop should be interrupted
            # the other callbacks are still being called.
            if response is False:
                logger.debug("An callback returned False. Abort is requested.")
                self._stop = True

        if save:
            self.save()
