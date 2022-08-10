from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from smac.chooser.chooser import ConfigurationChooser
from smac.configspace import Configuration
from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.runhistory import RunInfo, RunValue, RunInfoIntent
from smac.intensification.parallel_scheduling import ParallelScheduler
from smac.intensification.successive_halving import _SuccessiveHalving
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class Hyperband(ParallelScheduler):
    """Races multiple challengers against an incumbent using Hyperband method.

    Implementation from "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)

    Hyperband is an extension of the Successive Halving intensifier. Please refer to `SuccessiveHalving` documentation
    for more detailed information about the different types of budgets possible and the way instances are handled.

    Internally, this class uses the _Hyperband private class which actually implements the hyperband logic.
    To allow for parallelism, Hyperband can create multiple _Hyperband instances, based on the number
    of idle workers available.

    Parameters
    ----------
    stats: smac.stats.stats.Stats
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
    """

    def __init__(
        self,
        scenario: Scenario,
        # instances: List[str],
        # instance_specifics: Mapping[str, str] = None,
        # algorithm_walltime_limit: Optional[float] = None,
        # deterministic: bool = False,
        # min_budget: Optional[float] = None,
        # max_budget: Optional[float] = None,
        eta: float = 3,
        # instance_order: str = "shuffle_once",
        min_challenger: int = 1,
        incumbent_selection: str = "highest_executed_budget",
        seed: int | None = None,
        n_seeds: int | None = None,
    ) -> None:

        super().__init__(
            scenario=scenario,
            # instances=instances,
            # instance_specifics=instance_specifics,
            # algorithm_walltime_limit=algorithm_walltime_limit,
            # deterministic=deterministic,
            min_challenger=min_challenger,
            seed=seed,
        )

        self.logger = get_logger(__name__)

        # Parameters for a new hyperband
        self.n_seeds = n_seeds
        self.instance_order = scenario.instance_order
        self.incumbent_selection = incumbent_selection
        self._instances = scenario.instances
        # self._instance_specifics = scenario.instance_specifics
        self.min_budget = scenario.min_budget
        self.max_budget = scenario.max_budget
        self.eta = eta

    def _get_intensifier_ranking(self, intensifier: AbstractIntensifier) -> Tuple[int, int]:
        """Given a intensifier, returns how advance it is. This metric will be used to determine
        what priority to assign to the intensifier.

        Parameters
        ----------
        intensifier: AbstractRacer
            Intensifier to rank based on run progress

        Returns
        -------
        ranking: int
            the higher this number, the faster the intensifier will get
            the running resources. For hyperband we can use the
            sh_intensifier stage, for example
        tie_breaker: int
            The configurations that have been launched to break ties. For
            example, in the case of Successive Halving it can be the number
            of configurations launched
        """
        # For mypy -- we expect to work with _Hyperband instances
        assert isinstance(intensifier, _Hyperband)
        assert intensifier.sh_intensifier

        # For hyperband, we use the internal successive halving as a criteria
        # to see how advanced this intensifier is
        stage = 0
        if hasattr(intensifier.sh_intensifier, "stage"):
            # Newly created SuccessiveHalving objects have no stage
            stage = intensifier.sh_intensifier.stage

        return stage, len(intensifier.sh_intensifier.run_tracker)

    def _add_new_instance(self, num_workers: int) -> bool:
        """Decides if it is possible to add a new intensifier instance, and adds it. If a new
        intensifier instance is added, True is returned, else False.

        Parameters
        ----------
        num_workers: int
            the maximum number of workers available at a given time.

        Returns
        -------
            Whether or not a new instance was added.
        """
        if len(self.intensifier_instances) >= num_workers:
            return False

        hp = _Hyperband(
            scenario=self.scenario,
            # instances=self._instances,
            # instance_specifics=self._instance_specifics,
            # algorithm_walltime_limit=self.algorithm_walltime_limit,
            # deterministic=self.deterministic,
            # min_budget=self.min_budget,
            # max_budget=self.max_budget,
            eta=self.eta,
            # instance_order=self.instance_order,
            min_challenger=self.min_challenger,
            incumbent_selection=self.incumbent_selection,
            identifier=len(self.intensifier_instances),
            seed=self.seed,
            n_seeds=self.n_seeds,
        )
        hp._set_stats(self.stats)
        self.intensifier_instances[len(self.intensifier_instances)] = hp

        return True


class _Hyperband(_SuccessiveHalving):
    """Races multiple challengers against an incumbent using Hyperband method.

    This class contains the logic to implement:
    "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)

    Objects from this class are meant to run on a single worker. `Hyperband` method,
    creates multiple _Hyperband instances to allow parallelism, and for this reason
    `Hyperband` should be considered the user interface whereas `_Hyperband` a private
    class with the actual implementation of the method.

    Parameters
    ----------
    stats: smac.stats.stats.Stats
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
        scenario: Scenario,
        # instances: List[str],
        # instance_specifics: Mapping[str, str] = None,
        # algorithm_walltime_limit: Optional[float] = None,
        # deterministic: bool = False,
        # min_budget: Optional[float] = None,
        # max_budget: Optional[float] = None,
        eta: float = 3,
        # instance_order: str = "shuffle_once",
        min_challenger: int = 1,
        incumbent_selection: str = "highest_executed_budget",
        identifier: int = 0,
        seed: int | None = None,
        n_seeds: int | None = None,
    ) -> None:

        super().__init__(
            scenario=scenario,
            # instances=instances,
            # instance_specifics=instance_specifics,
            # algorithm_walltime_limit=algorithm_walltime_limit,
            # deterministic=deterministic,
            # min_budget=min_budget,
            # max_budget=max_budget,
            eta=eta,
            num_initial_challengers=None,  # Initial challengers passed as None
            # instance_order=instance_order,
            min_challenger=min_challenger,
            incumbent_selection=incumbent_selection,
            seed=seed,
            n_seeds=n_seeds,
        )

        self.identifier = identifier
        self.logger = get_logger(f"{__name__}.{identifier}")

        # To track completed hyperband iterations
        self.hb_iters = 0
        self.sh_intensifier: _SuccessiveHalving | None = None

    def process_results(
        self,
        run_info: RunInfo,
        incumbent: Optional[Configuration],
        runhistory: RunHistory,
        time_bound: float,
        result: RunValue,
        log_traj: bool = True,
    ) -> Tuple[Configuration, float]:
        """The intensifier stage will be updated based on the results/status of a configuration
        execution. Also, a incumbent will be determined.

        Parameters
        ----------
        run_info : RunInfo
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
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        incumbent: Configuration
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """
        assert self.sh_intensifier

        # run 1 iteration of successive halving
        incumbent, inc_perf = self.sh_intensifier.process_results(
            run_info=run_info,
            incumbent=incumbent,
            runhistory=runhistory,
            time_bound=time_bound,
            result=result,
            log_traj=log_traj,
        )
        self.run_id += 1

        # reset if SH iteration is over, else update for next iteration
        if self.sh_intensifier.iteration_done:
            self._update_stage()

        return incumbent, inc_perf

    def get_next_run(
        self,
        challengers: Optional[List[Configuration]],
        incumbent: Configuration,
        chooser: Optional[ConfigurationChooser],
        runhistory: RunHistory,
        repeat_configs: bool = True,
        num_workers: int = 1,
    ) -> Tuple[RunInfoIntent, RunInfo]:
        """Selects which challenger to use based on the iteration stage and set the iteration
        parameters. First iteration will choose configurations from the ``chooser`` or input
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
        num_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
        intent: RunInfoIntent
            Indicator of how to consume the RunInfo object
        run_info: RunInfo
            An object that encapsulates necessary information for a config run
        """
        if num_workers > 1:
            raise ValueError(
                "HyperBand does not support more than 1 worker, yet "
                "the argument num_workers to get_next_run is {}".format(num_workers)
            )

        if not hasattr(self, "s"):
            # initialize tracking variables
            self._update_stage()

        # sampling from next challenger marks the beginning of a new iteration
        self.iteration_done = False

        assert self.sh_intensifier
        intent, run_info = self.sh_intensifier.get_next_run(
            challengers=challengers,
            incumbent=incumbent,
            chooser=chooser,
            runhistory=runhistory,
            repeat_configs=self.sh_intensifier.repeat_configs,
        )

        # For testing purposes, this attribute highlights whether a
        # new challenger is proposed or not. Not required from a functional
        # perspective
        self.new_challenger = self.sh_intensifier.new_challenger

        return intent, run_info

    def _update_stage(self, runhistory: RunHistory = None) -> None:
        """Update tracking information for a new stage/iteration and update statistics. This method
        is called to initialize stage variables and after all configurations of a successive halving
        stage are completed.

        Parameters
        ----------
         runhistory : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        """
        if not hasattr(self, "s"):
            # setting initial running budget for future iterations (s & s_max from Algorithm 1)
            self.s_max = int(np.floor(np.log(self.max_budget / self.min_budget) / np.log(self.eta)))
            self.s = self.s_max
        elif self.s == 0:
            # reset if HB iteration is over
            self.s = self.s_max
            self.hb_iters += 1
            self.iteration_done = True
            self.run_id = 0
        else:
            # update for next iteration
            self.s -= 1

        # compute min budget for new SH run
        sh_min_budget = self.eta**-self.s * self.max_budget
        # sample challengers for next iteration (based on HpBandster package)
        n_challengers = int(np.floor((self.s_max + 1) / (self.s + 1)) * self.eta**self.s)

        # Compute this for the next round
        n_configs_in_stage = n_challengers * np.power(self.eta, -np.linspace(0, self.s, self.s + 1))
        n_configs_in_stage = np.array(np.round(n_configs_in_stage), dtype=int).tolist()

        self.logger.info(
            "Finished Hyperband iteration-step %d-%d with initial budget %d."
            % (self.hb_iters + 1, self.s_max - self.s + 1, sh_min_budget)
        )

        # Creating a new Successive Halving intensifier with the current running budget
        self.sh_intensifier = _SuccessiveHalving(
            scenario=self.scenario,
            # instances=self.instances,
            # instance_specifics=self.instance_specifics,
            # algorithm_walltime_limit=self.algorithm_walltime_limit,
            # deterministic=self.deterministic,
            # min_budget=sh_min_budget,
            # max_budget=self.max_budget,
            eta=self.eta,
            num_initial_challengers=n_challengers,
            # instance_order=self.instance_order,
            instance_seed_pairs=self.instance_seed_pairs,  # Additional argument to avoid
            identifier=self.identifier,
            seed=self.seed,
            n_seeds=self.n_seeds,
            _all_budgets=self.all_budgets[(-self.s - 1) :],
            _n_configs_in_stage=n_configs_in_stage,
        )
        self.sh_intensifier._set_stats(self.stats)
