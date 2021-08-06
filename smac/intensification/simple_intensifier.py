import typing

import numpy as np

from smac.configspace import Configuration
from smac.intensification.abstract_racer import AbstractRacer, RunInfoIntent
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.runhistory.runhistory import RunHistory, RunInfo, RunValue
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT
from smac.utils.io.traj_logging import TrajLogger

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class SimpleIntensifier(AbstractRacer):

    """ Performs the traditional Bayesian Optimization loop, without
        instance/seed intensification

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
        cutoff of TA runs
    deterministic : bool
        whether the TA is deterministic or not
    run_obj_time : bool
        whether the run objective is runtime or not (if true, apply adaptive capping)
    """

    def __init__(self,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 cutoff: typing.Optional[float] = None,
                 deterministic: bool = False,
                 run_obj_time: bool = True,
                 **kwargs: typing.Any
                 ) -> None:

        super().__init__(stats=stats,
                         traj_logger=traj_logger,
                         rng=rng,
                         instances=instances,
                         instance_specifics=instance_specifics,
                         cutoff=cutoff,
                         deterministic=deterministic,
                         run_obj_time=run_obj_time,
                         adaptive_capping_slackfactor=1.0,
                         min_chall=1,
                         )

        # We want to control the number of runs that are sent to
        # the workers. At any time, we want to make sure that if there
        # are just W workers, there should be at max W active runs
        # Below variable tracks active runs not processed
        self.run_tracker = {}  # type: typing.Dict[typing.Tuple[Configuration, str, int, float], bool]

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
        # Mark the fact that we processed this configuration
        self.run_tracker[(run_info.config, run_info.instance, run_info.seed, run_info.budget)] = True

        # If The incumbent is None we use the challenger
        if not incumbent:
            self.logger.info(
                "First run, no incumbent provided; challenger is assumed to be the incumbent"
            )
            incumbent = run_info.config

        self.num_run += 1

        incumbent = self._compare_configs(challenger=run_info.config,
                                          incumbent=incumbent,
                                          run_history=run_history,
                                          log_traj=log_traj)
        # get incumbent cost
        inc_perf = run_history.get_cost(incumbent)

        return incumbent, inc_perf

    def get_next_run(self,
                     challengers: typing.Optional[typing.List[Configuration]],
                     incumbent: Configuration,
                     chooser: typing.Optional[EPMChooser],
                     run_history: RunHistory,
                     repeat_configs: bool = True,
                     num_workers: int = 1,
                     ) -> typing.Tuple[RunInfoIntent, RunInfo]:
        """
        Selects which challenger to be used. As in a traditional BO loop,
        we sample from the EPM, which is the next configuration based on
        the acquisition function. The input data is read from the runhistory.

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
        intent: RunInfoIntent
               Indicator of how to consume the RunInfo object
        run_info: RunInfo
            An object that encapsulates the minimum information to
            evaluate a configuration
         """
        # We always sample from the configs provided or the EPM
        challenger = self._next_challenger(challengers=challengers,
                                           chooser=chooser,
                                           run_history=run_history,
                                           repeat_configs=repeat_configs)

        # Run tracker is a dictionary whose values indicate if a run has been
        # processed. If a value in this dict is false, it means that a worker
        # should still be processing this configuration.
        total_active_runs = len([v for v in self.run_tracker.values() if not v])
        if total_active_runs >= num_workers:
            # We only submit jobs if there is an idle worker
            return RunInfoIntent.WAIT, RunInfo(
                config=None,
                instance=None,
                instance_specific="0",
                seed=0,
                cutoff=self.cutoff,
                capped=False,
                budget=0.0,
            )

        run_info = RunInfo(
            config=challenger,
            instance=self.instances[-1],
            instance_specific="0",
            seed=0 if self.deterministic else int(self.rs.randint(low=0, high=MAXINT, size=1)[0]),
            cutoff=self.cutoff,
            capped=False,
            budget=0.0,
        )

        self.run_tracker[(run_info.config, run_info.instance, run_info.seed, run_info.budget)] = False
        return RunInfoIntent.RUN, run_info
