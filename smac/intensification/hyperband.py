import logging
import typing

import numpy as np

from smac.intensification.successive_halving import SuccessiveHalving
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory
from smac.tae.execute_ta_run import ExecuteTARun
from smac.utils.io.traj_logging import TrajLogger

__author__ = "Ashwin Raaghav Narayanan"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class Hyperband(SuccessiveHalving):
    """ Races multiple challengers against an incumbent using Hyperband method

    Implementation from "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)

    Parameters
    ----------
    tae_runner : tae.executre_ta_run_*.ExecuteTARun* Object
        target algorithm run executor
    stats: Stats
        stats object
    traj_logger: TrajLogger
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
    """

    def __init__(self, tae_runner: ExecuteTARun,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 cutoff: typing.Optional[int] = None,
                 deterministic: bool = False,
                 initial_budget: typing.Optional[float] = None,
                 max_budget: typing.Optional[float] = None,
                 eta: float = 3,
                 run_obj_time: bool = True,
                 n_seeds: typing.Optional[int] = None,
                 instance_order='shuffle_once',
                 adaptive_capping_slackfactor: float = 1.2,
                 **kwargs):

        super().__init__(tae_runner, stats, traj_logger, rng, instances,
                         instance_specifics, cutoff, deterministic,
                         initial_budget, max_budget, eta, None,  # initial challengers passed as None
                         run_obj_time, n_seeds, instance_order,
                         adaptive_capping_slackfactor, **kwargs)

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        # to track completed hyperband iterations
        self.hb_iters = 0

        # hyperband configuration
        self._init_hb_params()

    def _init_hb_params(self) -> None:
        """
        initialize Hyperband related parameters
        """
        # setting initial running budget for future iterations (s & s_max from Algorithm 1)
        self.s_max = np.floor(np.log(self.max_budget / self.initial_budget) / np.log(self.eta))
        self.s = np.floor(np.log(self.max_budget / self.initial_budget) / np.log(self.eta))

    def intensify(self, challengers: typing.List[Configuration],
                  incumbent: typing.Optional[Configuration],
                  run_history: RunHistory,
                  aggregate_func: typing.Callable,
                  time_bound: float = float(MAXINT),
                  log_traj: bool = True) -> typing.Tuple[Configuration, float]:
        """
        Running intensification via hyperband to determine the incumbent configuration.
        *Side effect:* adds runs to run_history

        Implementation of hyperband (Li et al., 2017)

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        incumbent : Configuration
            best configuration so far
        run_history : RunHistory
            stores all runs we ran so far
        aggregate_func: typing.Callable
            aggregate error across instances
        time_bound : float, optional (default=2 ** 31 - 1)
            time in [sec] available to perform intensify
        log_traj: bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        Configuration
            new incumbent configuration
        float
            empirical performance of incumbent configuration
        """
        # NOTE Since hyperband requires sampling for new configurations between its iterations,
        #      the intensification is spread across multiple intensify calls

        # compute min budget for new SH run
        sh_initial_budget = self.eta**-self.s * self.max_budget
        # sample challengers for next iteration
        # NOTE from HpBandster package
        n_challengers = int(np.floor((self.s_max+1) / (self.s + 1)) * self.eta**self.s)

        self.logger.info('Hyperband iteration-step: %d-%d  with initial budget: %d' % (
            self.hb_iters+1, self.s_max-self.s+1, sh_initial_budget))

        # creating a new Successive Halving intensifier with the current running budget
        sh_intensifier = SuccessiveHalving(self.tae_runner, self.stats, self.traj_logger, self.rs, self.instances,
                                           self.instance_specifics, self.cutoff, self.deterministic,
                                           sh_initial_budget, self.max_budget, self.eta, n_challengers,
                                           self.run_obj_time, self.n_seeds, self.instance_order,
                                           self.adaptive_capping_slackfactor)

        # run 1 iteration of successive halving
        incumbent, inc_perf = sh_intensifier.intensify(challengers=challengers,
                                                       incumbent=incumbent,
                                                       run_history=run_history,
                                                       aggregate_func=aggregate_func,
                                                       time_bound=time_bound, log_traj=log_traj)

        # reset if hyperband iteration is over, else update for next iteration
        if self.s == 0:
            self.s = self.s_max
            self.hb_iters += 1
        else:
            self.s -= 1

        return incumbent, inc_perf
