from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from smac.runhistory import TrialInfo, TrialInfoIntent, TrialValue
from smac.runhistory.runhistory import RunHistory
from smac.utils.logging import get_logger
from smac.intensification.hyperband import Hyperband
from smac.intensification.successive_halving_worker import SuccessiveHalvingWorker
from ConfigSpace import Configuration

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class HyperbandWorker(SuccessiveHalvingWorker):
    """Races multiple challengers against an incumbent using Hyperband method.

    This class contains the logic to implement:
    "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)

    Objects from this class are meant to run on a single worker. `Hyperband` method,
    creates multiple _Hyperband instances to allow parallelism, and for this reason
    `Hyperband` should be considered the user interface whereas `_Hyperband` a private
    class with the actual implementation of the method.

    Parameters
    ----------
    stats: smac._stats._stats._stats
        stats object
    rng : np.random.RandomState
    instances : List[str]
        list of all instance ids
    instance_specifics : Mapping[str, str]
        mapping from instance name to instance specific string
    algorithm_walltime_limit : Optional[int]
        runtime algorithm_walltime_limit of TA runs
    deterministic : bool
        whether the TA is deterministic or not
    min_budget : Optional[float]
        minimum budget allowed for 1 run of successive halving
    max_budget : Optional[float]
        maximum budget allowed for 1 run of successive halving
    eta : float
        'halving' factor after each iteration in a successive halving run. Defaults to 3
    n_seeds : Optional[int]
        Number of seeds to use, if TA is not deterministic. Defaults to None, i.e., seed is set as 0
    instance_order : Optional[str]
        how to order instances. Can be set to: [None, shuffle_once, shuffle]
        * None - use as is given by the user
        * shuffle_once - shuffle once and use across all SH run (default)
        * shuffle - shuffle before every SH run
    min_challenger: int
        minimal number of challengers to be considered (even if time_bound is exhausted earlier). This class will
        raise an exception if a value larger than 1 is passed.
    incumbent_selection: str
        How to select incumbent in successive halving. Only active for real-valued budgets.
        Can be set to: [highest_executed_budget, highest_budget, any_budget]
        * highest_executed_budget - incumbent is the best in the highest budget run so far (default)
        * highest_budget - incumbent is selected only based on the highest budget
        * any_budget - incumbent is the best on any budget i.e., best performance regardless of budget
    identifier: int
        Allows to identify the _Hyperband instance in case of multiple ones
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
    def uses_seeds(self) -> bool:
        return self._hyperband.uses_seeds

    @property
    def uses_budgets(self) -> bool:
        return self._hyperband.uses_budgets

    @property
    def uses_instances(self) -> bool:
        return self._hyperband.uses_instances

    def process_results(
        self,
        trial_info: TrialInfo,
        trial_value: TrialValue,
        incumbent: Configuration | None,
        runhistory: RunHistory,
        time_bound: float,
        log_trajectory: bool = True,
    ) -> tuple[Configuration, float]:
        """The intensifier stage will be updated based on the results/status of a configuration
        execution. Also, a incumbent will be determined.

        Parameters
        ----------
        trial_info : RunInfo
            A RunInfo containing the configuration that was evaluated
        incumbent : Optional[Configuration]
            Best configuration seen so far
        runhistory : RunHistory
            stores all runs we ran so far
            if False, an evaluated configuration will not be generated again
        time_bound : float
            time in [sec] available to perform intensify
        result: RunValue
            Contain the result (status and other methadata) of exercising
            a challenger/incumbent.
        log_trajectory: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        incumbent: Configuration
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """
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

    def get_next_run(
        self,
        challengers: list[Configuration] | None,
        incumbent: Configuration,
        get_next_configurations: Callable[[], Iterator[Configuration]] | None,
        runhistory: RunHistory,
        repeat_configs: bool = True,
        n_workers: int = 1,
    ) -> tuple[TrialInfoIntent, TrialInfo]:
        """Selects which challenger to use based on the iteration stage and set the iteration
        parameters. First iteration will choose configurations from the ``ask`` or input
        challengers, while the later iterations pick top configurations from the previously selected
        challengers in that iteration.

        If no new run is available, the method returns a configuration of None.

        Parameters
        ----------
        challengers : List[Configuration]
            promising configurations
        incumbent: Configuration
               incumbent configuration
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing
        runhistory : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again
        n_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
        intent: RunInfoIntent
            Indicator of how to consume the RunInfo object
        trial_info: RunInfo
            An object that encapsulates necessary information for a config run
        """
        if n_workers > 1:
            raise ValueError(
                "HyperBand does not support more than 1 worker, yet "
                "the argument n_workers to get_next_run is {}".format(n_workers)
            )

        # Sampling from next challenger marks the beginning of a new iteration
        self._iteration_done = False

        assert self._sh_intensifier
        intent, trial_info = self._sh_intensifier.get_next_run(
            challengers=challengers,
            incumbent=incumbent,
            get_next_configurations=get_next_configurations,
            runhistory=runhistory,
            repeat_configs=self._sh_intensifier._repeat_configs,
        )

        # For testing purposes, this attribute highlights whether a
        # new challenger is proposed or not. Not required from a functional
        # perspective
        self._new_challenger = self._sh_intensifier._new_challenger

        return intent, trial_info

    def _update_stage(self, runhistory: RunHistory = None) -> None:
        """Update tracking information for a new stage/iteration and update statistics. This method
        is called to initialize stage variables and after all configurations of a successive halving
        stage are completed."""

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
