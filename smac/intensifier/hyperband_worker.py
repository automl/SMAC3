from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
from ConfigSpace import Configuration

from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving_worker import SuccessiveHalvingWorker
from smac.runhistory import TrialInfo, TrialInfoIntent, TrialValue
from smac.runhistory.runhistory import RunHistory
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class HyperbandWorker(SuccessiveHalvingWorker):
    """This is the worker class for Hyperband.

    Warning
    -------
    Do not use this class as stand-alone.

    Parameters
    ----------
    hyperband : Hyperband
        The controller of the instance.
    identifier : int, defaults to 0
        Adds a numerical identifier on the instance. Used for debug and tagging logger messages properly.
    """

    def __init__(
        self,
        hyperband: Hyperband,
        identifier: int = 0,
    ) -> None:

        super().__init__(
            successive_halving=hyperband,
            identifier=identifier,
        )

        min_budget = hyperband._min_budget
        max_budget = hyperband._max_budget
        eta = hyperband._eta

        # Overwrite logger
        self._identifier = identifier
        self._logger = get_logger(f"{__name__}.{identifier}")

        # To track completed hyperband iterations
        self._hyperband = hyperband
        self._hb_iters = 0
        self._sh_intensifier: SuccessiveHalvingWorker | None = None

        # Setting initial running budget for future iterations (s & s_max from Algorithm 1)
        self._s_max = int(np.floor(np.log(max_budget / min_budget) / np.log(eta)))
        self._s = self._s_max

        # We update our sh intensifier directly
        self._update_worker()

    @property
    def uses_seeds(self) -> bool:  # noqa: D102
        return self._hyperband.uses_seeds

    @property
    def uses_budgets(self) -> bool:  # noqa: D102
        return self._hyperband.uses_budgets

    @property
    def uses_instances(self) -> bool:  # noqa: D102
        return self._hyperband.uses_instances

    def process_results(
        self,
        trial_info: TrialInfo,
        trial_value: TrialValue,
        incumbent: Configuration | None,
        runhistory: RunHistory,
        time_bound: float,
        log_trajectory: bool = True,
    ) -> tuple[Configuration, float]:  # noqa: D102
        assert self._sh_intensifier

        # run 1 iteration of successive halving
        incumbent, inc_perf = self._sh_intensifier.process_results(
            trial_info=trial_info,
            trial_value=trial_value,
            incumbent=incumbent,
            runhistory=runhistory,
            time_bound=time_bound,
            log_trajectory=log_trajectory,
        )
        self._num_trials += 1

        # reset if SH iteration is over, else update for next iteration
        if self._sh_intensifier._iteration_done:
            self._update_stage()

        return incumbent, inc_perf

    def get_next_trial(
        self,
        challengers: list[Configuration] | None,
        incumbent: Configuration,
        get_next_configurations: Callable[[], Iterator[Configuration]] | None,
        runhistory: RunHistory,
        repeat_configs: bool = True,
        n_workers: int = 1,
    ) -> tuple[TrialInfoIntent, TrialInfo]:  # noqa: D102
        if n_workers > 1:
            raise ValueError(
                "HyperBand does not support more than 1 worker, yet "
                "the argument n_workers to get_next_trial is {}".format(n_workers)
            )

        # Sampling from next challenger marks the beginning of a new iteration
        self._iteration_done = False

        assert self._sh_intensifier
        intent, trial_info = self._sh_intensifier.get_next_trial(
            challengers=challengers,
            incumbent=incumbent,
            get_next_configurations=get_next_configurations,
            runhistory=runhistory,
            repeat_configs=self._sh_intensifier._repeat_configs,
        )

        # For testing purposes, this attribute highlights whether a new challenger is proposed or not
        # Not required from a functional perspective
        self._new_challenger = self._sh_intensifier._new_challenger

        return intent, trial_info

    def _update_stage(self, runhistory: RunHistory = None) -> None:
        if self._s == 0:
            # Reset if HB iteration is over
            self._s = self._s_max
            self._hb_iters += 1
            self._iteration_done = True
            self._num_trials = 0
        else:
            # Update for next iteration
            self._s -= 1

        self._update_worker()

    def _update_worker(self) -> None:
        """Updates the successive halving worker based on `self._s`."""
        max_budget = self._hyperband._max_budget
        eta = self._hyperband._eta

        # Compute min budget for new SH run
        sh_min_budget = eta**-self._s * max_budget

        # Sample challengers for next iteration (based on HpBandster package)
        n_challengers = int(np.floor((self._s_max + 1) / (self._s + 1)) * eta**self._s)

        # Compute this for the next round
        n_configs_in_stage = n_challengers * np.power(eta, -np.linspace(0, self._s, self._s + 1))
        n_configs_in_stage = np.array(np.round(n_configs_in_stage), dtype=int).tolist()

        self._logger.info(
            "Finished Hyperband iteration-step %d-%d with initial budget %d."
            % (self._hb_iters + 1, self._s_max - self._s + 1, sh_min_budget)
        )

        # Creating a new Successive Halving intensifier with the current running budget
        self._sh_intensifier = SuccessiveHalvingWorker(
            successive_halving=self._hyperband,
            identifier=self._identifier,
            _all_budgets=self._all_budgets[(-self._s - 1) :],
            _n_configs_in_stage=n_configs_in_stage,
            _min_budget=sh_min_budget,
            _max_budget=max_budget,
        )
        self._sh_intensifier._stats = self._stats
