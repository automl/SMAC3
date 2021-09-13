import logging
import typing
import time
from collections import OrderedDict
from enum import Enum

import numpy as np

from smac.optimizer.epm_configuration_chooser import EPMChooser

from smac.stats.stats import Stats
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory, RunInfo, RunValue
from smac.utils.io.traj_logging import TrajLogger

_config_to_run_type = typing.Iterator[typing.Optional[Configuration]]

__author__ = "Ashwin Raaghav Narayanan"
__copyright__ = "Copyright 2019, ML4AAD"
__license__ = "3-clause BSD"


class RunInfoIntent(Enum):
    """Class to define different requests on how to process
    the runinfo

    Gives the flexibility to indicate whether a configuration
    should be skipped (SKIP) or if the SMBO should simple
    run a generated run_info.
    """
    RUN = 0  # Normal run execution of a run info
    SKIP = 1  # Skip running the run_info
    WAIT = 2  # Wait for more configs to be processed


class AbstractRacer(object):
    """
    Base class for all racing methods

    The "intensification" is designed to output a RunInfo object with enough information
    to run a given configuration (for example, the run info contains the instance/seed
    pair, as well as the associated resources).

    A worker can execute this RunInfo object and produce a RunValue object with the
    execution results. Each intensifier process the RunValue object and updates it's
    internal state in preparation for the next iteration.

    **Note: Do not use directly**

    Parameters
    ----------
    stats: Stats
        stats object
    traj_logger: smac.utils.io.traj_logging.TrajLogger
        TrajLogger object to log all new incumbents
    rng : np.random.RandomState
    instances : typing.List[str]
        list of all instance ids
    instance_specifics : typing.Mapping[str,np.ndarray]
        mapping from instance name to instance specific string
    cutoff : float
        runtime cutoff of TA runs
    deterministic: bool
        whether the TA is deterministic or not
    run_obj_time: bool
        whether the run objective is runtime or not (if true, apply adaptive capping)
    minR : int
        Minimum number of run per config (summed over all calls to
        intensify).
    maxR : int
        Maximum number of runs per config (summed over all calls to
        intensifiy).
    adaptive_capping_slackfactor: float
        slack factor of adpative capping (factor * adpative cutoff)
    min_chall: int
        minimal number of challengers to be considered (even if time_bound is exhausted earlier)
    """

    def __init__(self,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Optional[typing.Mapping[str, np.ndarray]] = None,
                 cutoff: typing.Optional[float] = None,
                 deterministic: bool = False,
                 run_obj_time: bool = True,
                 minR: int = 1,
                 maxR: int = 2000,
                 adaptive_capping_slackfactor: float = 1.2,
                 min_chall: int = 1,):

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.stats = stats
        self.traj_logger = traj_logger
        self.rs = rng

        # scenario info
        self.cutoff = cutoff
        self.deterministic = deterministic
        self.run_obj_time = run_obj_time

        self.minR = minR
        self.maxR = maxR
        self.adaptive_capping_slackfactor = adaptive_capping_slackfactor
        self.min_chall = min_chall

        # instances
        if instances is None:
            instances = []
        # removing duplicates in the user provided instances
        self.instances = list(OrderedDict.fromkeys(instances))
        if instance_specifics is None:
            self.instance_specifics = {}  # type: typing.Mapping[str, np.ndarray]
        else:
            self.instance_specifics = instance_specifics

        # general attributes
        self.num_run = 0  # Number of runs done in an iteration so far
        self._chall_indx = 0
        self._ta_time = 0.

        # attributes for sampling next configuration
        # Repeating configurations is discouraged for parallel runs
        self.repeat_configs = False
        # to mark the end of an iteration
        self.iteration_done = False

    def get_next_run(self,
                     challengers: typing.Optional[typing.List[Configuration]],
                     incumbent: Configuration,
                     chooser: typing.Optional[EPMChooser],
                     run_history: RunHistory,
                     repeat_configs: bool = True,
                     num_workers: int = 1,
                     ) -> typing.Tuple[RunInfoIntent, RunInfo]:
        """
        Abstract method for choosing the next challenger, to allow for different selections across intensifiers
        uses ``_next_challenger()`` by default

        If no more challengers are available, the method should issue a SKIP via
        RunInfoIntent.SKIP, so that a new iteration can sample new configurations.

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
        num_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
        run_info: RunInfo
            An object that encapsulates necessary information for a config run
        intent: RunInfoIntent
            Indicator of how to consume the RunInfo object
        """
        raise NotImplementedError()

    def process_results(self,
                        run_info: RunInfo,
                        incumbent: typing.Optional[Configuration],
                        run_history: RunHistory,
                        time_bound: float,
                        result: RunValue,
                        log_traj: bool = True,
                        ) -> \
            typing.Tuple[Configuration, float]:
        """
        The intensifier stage will be updated based on the results/status
        of a configuration execution.
        Also, a incumbent will be determined.

        Parameters
        ----------
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated
        incumbent : typing.Optional[Configuration]
            Best configuration seen so far
        run_history : RunHistory
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
        incumbent: Configuration()
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """
        raise NotImplementedError()

    def _next_challenger(self,
                         challengers: typing.Optional[typing.List[Configuration]],
                         chooser: typing.Optional[EPMChooser],
                         run_history: RunHistory,
                         repeat_configs: bool = True) -> typing.Optional[Configuration]:
        """ Retuns the next challenger to use in intensification
        If challenger is None, then optimizer will be used to generate the next challenger

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations to evaluate next
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            a sampler that generates next configurations to use for racing
        run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again

        Returns
        -------
        Configuration
            next challenger to use
        """
        start_time = time.time()

        used_configs = set(run_history.get_all_configs())

        if challengers:
            # iterate over challengers provided
            self.logger.debug("Using challengers provided")
            chall_gen = (c for c in challengers)  # type: _config_to_run_type
        elif chooser:
            # generating challengers on-the-fly if optimizer is given
            self.logger.debug("Generating new challenger from optimizer")
            chall_gen = chooser.choose_next()
        else:
            raise ValueError('No configurations/chooser provided. Cannot generate challenger!')

        self.logger.debug('Time to select next challenger: %.4f' % (time.time() - start_time))

        # select challenger from the generators
        assert chall_gen is not None
        for challenger in chall_gen:
            # repetitions allowed
            if repeat_configs:
                return challenger

            # otherwise, select only a unique challenger
            if challenger not in used_configs:
                return challenger

        self.logger.debug("No valid challenger was generated!")
        return None

    def _adapt_cutoff(self,
                      challenger: Configuration,
                      run_history: RunHistory,
                      inc_sum_cost: float) -> float:
        """Adaptive capping:
        Compute cutoff based on time so far used for incumbent
        and reduce cutoff for next run of challenger accordingly

        !Only applicable if self.run_obj_time

        !runs on incumbent should be superset of the runs performed for the
         challenger

        Parameters
        ----------
        challenger : Configuration
            Configuration which challenges incumbent
        run_history : smac.runhistory.runhistory.RunHistory
            Stores all runs we ran so far
        inc_sum_cost: float
            Sum of runtimes of all incumbent runs

        Returns
        -------
        cutoff: float
            Adapted cutoff
        """

        if not self.run_obj_time:
            raise ValueError('This method only works when the run objective is time')

        curr_cutoff = self.cutoff if self.cutoff is not None else np.inf

        # cost used by challenger for going over all its runs
        # should be subset of runs of incumbent (not checked for efficiency
        # reasons)
        chall_inst_seeds = run_history.get_runs_for_config(challenger, only_max_observed_budget=True)
        chal_sum_cost = run_history.sum_cost(
            config=challenger,
            instance_seed_budget_keys=chall_inst_seeds,
        )
        cutoff = min(curr_cutoff,
                     inc_sum_cost * self.adaptive_capping_slackfactor - chal_sum_cost
                     )
        return cutoff

    def _compare_configs(self,
                         incumbent: Configuration,
                         challenger: Configuration,
                         run_history: RunHistory,
                         log_traj: bool = True) -> typing.Optional[Configuration]:
        """
        Compare two configuration wrt the runhistory and return the one which
        performs better (or None if the decision is not safe)

        Decision strategy to return x as being better than y:
            1. x has at least as many runs as y
            2. x performs better than y on the intersection of runs on x and y

        Implicit assumption:
            Challenger was evaluated on the same instance-seed pairs as
            incumbent

        Parameters
        ----------
        incumbent: Configuration
            Current incumbent
        challenger: Configuration
            Challenger configuration
        run_history: smac.runhistory.runhistory.RunHistory
            Stores all runs we ran so far
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        None or better of the two configurations x,y
        """

        inc_runs = run_history.get_runs_for_config(incumbent, only_max_observed_budget=True)
        chall_runs = run_history.get_runs_for_config(challenger, only_max_observed_budget=True)
        to_compare_runs = set(inc_runs).intersection(chall_runs)

        # performance on challenger runs
        chal_perf = run_history.average_cost(challenger, to_compare_runs)
        inc_perf = run_history.average_cost(incumbent, to_compare_runs)

        # Line 15
        if chal_perf > inc_perf and len(chall_runs) >= self.minR:
            # Incumbent beats challenger
            self.logger.debug("Incumbent (%.4f) is better than challenger "
                              "(%.4f) on %d runs." %
                              (inc_perf, chal_perf, len(chall_runs)))
            return incumbent

        # Line 16
        if not set(inc_runs) - set(chall_runs):

            # no plateau walks
            if chal_perf >= inc_perf:
                self.logger.debug("Incumbent (%.4f) is at least as good as the "
                                  "challenger (%.4f) on %d runs." %
                                  (inc_perf, chal_perf, len(chall_runs)))
                if log_traj and self.stats.inc_changed == 0:
                    # adding incumbent entry
                    self.stats.inc_changed += 1  # first incumbent
                    self.traj_logger.add_entry(train_perf=inc_perf,
                                               incumbent_id=self.stats.inc_changed,
                                               incumbent=incumbent)
                return incumbent

            # Challenger is better than incumbent
            # and has at least the same runs as inc
            # -> change incumbent
            n_samples = len(chall_runs)
            self.logger.info("Challenger (%.4f) is better than incumbent (%.4f)"
                             " on %d runs." % (chal_perf, inc_perf, n_samples))
            self._log_incumbent_changes(incumbent, challenger)

            if log_traj:
                self.stats.inc_changed += 1
                self.traj_logger.add_entry(train_perf=chal_perf,
                                           incumbent_id=self.stats.inc_changed,
                                           incumbent=challenger)
            return challenger

        # undecided
        return None

    def _log_incumbent_changes(
        self,
        incumbent: Configuration,
        challenger: Configuration,
    ) -> None:
        params = sorted([(param, incumbent[param], challenger[param]) for param in challenger.keys()])
        self.logger.info("Changes in incumbent:")
        for param in params:
            if param[1] != param[2]:
                self.logger.info("  %s : %r -> %r" % param)
            else:
                self.logger.debug("  %s remains unchanged: %r", param[0], param[1])
