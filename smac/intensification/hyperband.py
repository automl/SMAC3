import logging
import typing

import numpy as np

from smac.intensification.successive_halving import SuccessiveHalving
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.stats.stats import Stats
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory import RunValue, RunInfo, StatusType  # noqa: F401
from smac.utils.io.traj_logging import TrajLogger

__author__ = "Ashwin Raaghav Narayanan"
__copyright__ = "Copyright 2019, ML4AAD"
__license__ = "3-clause BSD"


class Hyperband(SuccessiveHalving):
    """ Races multiple challengers against an incumbent using Hyperband method

    Implementation from "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)

    Hyperband is an extension of the Successive Halving intensifier. Please refer to `SuccessiveHalving` documentation
    for more detailed information about the different types of budgets possible and the way instances are handled.

    Parameters
    ----------
    stats: smac.stats.stats.Stats
        stats object
    traj_logger: smac.utils.io.traj_logging.TrajLogger
        TrajLogger object to log all new incumbents
    rng : np.random.RandomState
    instances : typing.List[str]
        list of all instance ids
    instance_specifics : typing.Mapping[str,np.ndarray]
        mapping from instance name to instance specific string
    cutoff : typing.Optional[int]
        runtime cutoff of TA runs
    deterministic : bool
        whether the TA is deterministic or not
    initial_budget : typing.Optional[float]
        minimum budget allowed for 1 run of successive halving
    max_budget : typing.Optional[float]
        maximum budget allowed for 1 run of successive halving
    eta : float
        'halving' factor after each iteration in a successive halving run. Defaults to 3
    run_obj_time : bool
        whether the run objective is runtime or not (if true, apply adaptive capping)
    n_seeds : typing.Optional[int]
        Number of seeds to use, if TA is not deterministic. Defaults to None, i.e., seed is set as 0
    instance_order : typing.Optional[str]
        how to order instances. Can be set to: [None, shuffle_once, shuffle]
        * None - use as is given by the user
        * shuffle_once - shuffle once and use across all SH run (default)
        * shuffle - shuffle before every SH run
    adaptive_capping_slackfactor : float
        slack factor of adpative capping (factor * adpative cutoff)
    min_chall: int
        minimal number of challengers to be considered (even if time_bound is exhausted earlier). This class will
        raise an exception if a value larger than 1 is passed.
    incumbent_selection: str
        How to select incumbent in successive halving. Only active for real-valued budgets.
        Can be set to: [highest_executed_budget, highest_budget, any_budget]
        * highest_executed_budget - incumbent is the best in the highest budget run so far (default)
        * highest_budget - incumbent is selected only based on the highest budget
        * any_budget - incumbent is the best on any budget i.e., best performance regardless of budget
    """

    def __init__(self,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 cutoff: typing.Optional[float] = None,
                 deterministic: bool = False,
                 initial_budget: typing.Optional[float] = None,
                 max_budget: typing.Optional[float] = None,
                 eta: float = 3,
                 run_obj_time: bool = True,
                 n_seeds: typing.Optional[int] = None,
                 instance_order: str = 'shuffle_once',
                 adaptive_capping_slackfactor: float = 1.2,
                 min_chall: int = 1,
                 incumbent_selection: str = 'highest_executed_budget',
                 ) -> None:

        super().__init__(stats=stats,
                         traj_logger=traj_logger,
                         rng=rng,
                         instances=instances,
                         instance_specifics=instance_specifics,
                         cutoff=cutoff,
                         deterministic=deterministic,
                         initial_budget=initial_budget,
                         max_budget=max_budget,
                         eta=eta,
                         num_initial_challengers=None,  # initial challengers passed as None
                         run_obj_time=run_obj_time,
                         n_seeds=n_seeds,
                         instance_order=instance_order,
                         adaptive_capping_slackfactor=adaptive_capping_slackfactor,
                         min_chall=min_chall,
                         incumbent_selection=incumbent_selection,)

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        # to track completed hyperband iterations
        self.hb_iters = 0
        self.sh_intensifier = None  # type: SuccessiveHalving # type: ignore[assignment]

    def process_results(self,
                        challenger: Configuration,
                        incumbent: typing.Optional[Configuration],
                        run_history: RunHistory,
                        time_bound: float,
                        result: RunValue,
                        log_traj: bool = True,
                        ) -> \
            typing.Tuple[typing.Optional[Configuration], float]:
        """
        The intensifier stage will be updated based on the results/status
        of a configuration execution.
        Also, a incumbent will be determined.

        Parameters
        ----------
        challenger : Configuration
            A configuration that was previously executed, and whose status
            will be used to define the next stage.
        incumbet : Configuration
            Best configuration seen so far
        run_history : typing.Optional[smac.runhistory.runhistory.RunHistory]
            stores all runs we ran so far
            if False, an evaluated configuration will not be generated again
        time_bound : float, optional (default=2 ** 31 - 1)
            time in [sec] available to perform intensify
        result: RunValue
            Contain the result (status and other methadata) of exercising
            a challenger/incumbent.
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        incumbent: Configuration()
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """

        # run 1 iteration of successive halving
        incumbent, inc_perf = self.sh_intensifier.process_results(challenger=challenger,
                                                                  incumbent=incumbent,
                                                                  run_history=run_history,
                                                                  time_bound=time_bound,
                                                                  result=result,
                                                                  log_traj=log_traj)
        self.num_run += 1

        # reset if SH iteration is over, else update for next iteration
        if self.sh_intensifier.iteration_done:
            self._update_stage()

        return incumbent, inc_perf

    def get_next_run(self,
                     challengers: typing.Optional[typing.List[Configuration]],
                     incumbent: Configuration,
                     chooser: typing.Optional[EPMChooser],
                     run_history: RunHistory,
                     repeat_configs: bool = True) -> RunInfo:
        """
        Selects which challenger to use based on the iteration stage and set the iteration parameters.
        First iteration will choose configurations from the ``chooser`` or input challengers,
        while the later iterations pick top configurations from the previously selected challengers in that iteration

        If no new run is available, the method returns a configuration of None.

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        incumbent: Configuration
               incumbent configuration
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing
        run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again

        Returns
        -------
        run_info: RunInfo
               An object that encapsulates necessary information for a config run
        """

        if not hasattr(self, 's'):
            # initialize tracking variables
            self._update_stage()

        # sampling from next challenger marks the beginning of a new iteration
        self.iteration_done = False

        run_info = self.sh_intensifier.get_next_run(
            challengers=challengers,
            incumbent=incumbent,
            chooser=chooser,
            run_history=run_history,
            repeat_configs=self.sh_intensifier.repeat_configs
        )

        # For testing purposes, this attribute highlights whether a
        # new challenger is proposed or not. Not required from a functional
        # perspective
        self.new_challenger = self.sh_intensifier.new_challenger

        return run_info

    def _update_stage(self, run_history: RunHistory = None) -> None:
        """
        Update tracking information for a new stage/iteration and update statistics.
        This method is called to initialize stage variables and after all configurations
        of a successive halving stage are completed.

        Parameters
        ----------
         run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        """
        if not hasattr(self, 's'):
            # setting initial running budget for future iterations (s & s_max from Algorithm 1)
            self.s_max = int(np.floor(np.log(self.max_budget / self.initial_budget) / np.log(self.eta)))
            self.s = self.s_max
        elif self.s == 0:
            # reset if HB iteration is over
            self.s = self.s_max
            self.hb_iters += 1
            self.iteration_done = True
            self.num_run = 0
        else:
            # update for next iteration
            self.s -= 1

        # compute min budget for new SH run
        sh_initial_budget = self.eta ** -self.s * self.max_budget
        # sample challengers for next iteration (based on HpBandster package)
        n_challengers = int(np.floor((self.s_max + 1) / (self.s + 1)) * self.eta ** self.s)

        self.logger.info('Hyperband iteration-step: %d-%d  with initial budget: %d' % (
            self.hb_iters + 1, self.s_max - self.s + 1, sh_initial_budget))

        # creating a new Successive Halving intensifier with the current running budget
        self.sh_intensifier = SuccessiveHalving(
            stats=self.stats,
            traj_logger=self.traj_logger,
            rng=self.rs,
            instances=self.instances,
            instance_specifics=self.instance_specifics,
            cutoff=self.cutoff,
            deterministic=self.deterministic,
            initial_budget=sh_initial_budget,
            max_budget=self.max_budget,
            eta=self.eta,
            num_initial_challengers=n_challengers,
            run_obj_time=self.run_obj_time,
            n_seeds=self.n_seeds,
            instance_order=self.instance_order,
            adaptive_capping_slackfactor=self.adaptive_capping_slackfactor,
            inst_seed_pairs=self.inst_seed_pairs  # additional argument to avoid
        )  # processing instances & seeds again
