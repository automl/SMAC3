import logging
import typing

import numpy as np

from smac.intensification.abstract_racer import AbstractRacer
from smac.intensification.hyperband import Hyperband
from smac.intensification.parallel_scheduling import ParallelScheduler
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger


class ParallelHyperband(ParallelScheduler):
    """ Races multiple challengers against an incumbent using Hyperband method,
    with iterations that run over multiple workers.

    This class can be considered a wrapper over Hyperband, that creates multiple
    intensifiers to prevent having idle workers

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
                         run_obj_time=run_obj_time,
                         adaptive_capping_slackfactor=adaptive_capping_slackfactor,
                         min_chall=min_chall)

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        # Parameters for a new hyperband
        self.n_seeds = n_seeds
        self.instance_order = instance_order
        self.incumbent_selection = incumbent_selection
        self._instances = instances
        self._instance_specifics = instance_specifics
        self.initial_budget = initial_budget
        self.max_budget = max_budget
        self.eta = eta

    def _get_intensifier_ranking(self, intensifier: AbstractRacer
                                 ) -> typing.Tuple[int, int]:
        """
        Given a intensifier, returns how advance it is.
        This metric will be used to determine what priority to
        assign to the intensifier

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
        # For mypy -- we expect to work with Hyperband instances
        assert isinstance(intensifier, Hyperband)

        # For hyperband, we use the internal successive halving as a criteria
        # to see how advanced this intensifier is
        stage = 0
        if hasattr(intensifier.sh_intensifier, 'stage'):
            # Newly created SuccessiveHalving objects have no stage
            stage = intensifier.sh_intensifier.stage
        return stage, len(intensifier.sh_intensifier.run_tracker)

    def _add_new_instance(self, num_workers: int) -> bool:
        """
        Decides if it is possible to add a new intensifier instance,
        and adds it.
        If a new intensifier instance is added, True is returned, else False.

        Parameters:
        -----------
        num_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
            Whether or not a new instance was added
        """
        if len(self.intensifier_instances) >= num_workers:
            return False

        self.intensifier_instances[len(self.intensifier_instances)] = Hyperband(
            stats=self.stats,
            traj_logger=self.traj_logger,
            rng=self.rs,
            instances=self._instances,
            instance_specifics=self._instance_specifics,
            cutoff=self.cutoff,
            deterministic=self.deterministic,
            initial_budget=self.initial_budget,
            max_budget=self.max_budget,
            eta=self.eta,
            run_obj_time=self.run_obj_time,
            n_seeds=self.n_seeds,
            instance_order=self.instance_order,
            adaptive_capping_slackfactor=self.adaptive_capping_slackfactor,
            min_chall=self.min_chall,
            incumbent_selection=self.incumbent_selection,
            identifier=len(self.intensifier_instances),
        )

        return True
