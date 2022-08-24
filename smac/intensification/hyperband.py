from __future__ import annotations

from typing import Any, Tuple

from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.intensification.parallel_scheduling import ParallelScheduler
from smac.intensification.hyperband_worker import HyperbandWorker
from smac.scenario import Scenario

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
        instance_seed_pairs: list[tuple[str, int]] | None = None,
        instance_order: str | None = "shuffle_once",
        incumbent_selection: str = "highest_executed_budget",
        min_challenger: int = 1,
        eta: float = 3,
        intensify_percentage: float = 0.5,
        seed: int | None = None,
        n_seeds: int | None = None,
    ) -> None:

        super().__init__(
            scenario=scenario,
            min_challenger=min_challenger,
            intensify_percentage=intensify_percentage,
            seed=seed,
        )

        self.instance_seed_pairs = instance_seed_pairs
        self.instance_order = instance_order
        self.incumbent_selection = incumbent_selection
        self.eta = eta
        self.n_seeds = n_seeds

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

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
        assert isinstance(intensifier, HyperbandWorker)
        assert intensifier.sh_intensifier

        # For hyperband, we use the internal successive halving as a criteria
        # to see how advanced this intensifier is
        stage = 0
        if hasattr(intensifier.sh_intensifier, "stage"):
            # Newly created SuccessiveHalving objects have no stage
            stage = intensifier.sh_intensifier.stage

        return stage, len(intensifier.sh_intensifier.run_tracker)

    def _add_new_instance(self, n_workers: int) -> bool:
        """Decides if it is possible to add a new intensifier instance, and adds it. If a new
        intensifier instance is added, True is returned, else False.

        Parameters
        ----------
        n_workers: int
            the maximum number of workers available at a given time.

        Returns
        -------
            Whether or not a new instance was added.
        """
        if len(self.intensifier_instances) >= n_workers:
            return False

        hp = HyperbandWorker(
            scenario=self.scenario,
            # instances=self._instances,
            # instance_specifics=self._instance_specifics,
            # algorithm_walltime_limit=self.algorithm_walltime_limit,
            # deterministic=self.deterministic,
            # min_budget=self.min_budget,
            # max_budget=self.max_budget,
            eta=self.eta,
            instance_seed_pairs=self.instance_seed_pairs,
            instance_order=self.instance_order,
            min_challenger=self.min_challenger,
            incumbent_selection=self.incumbent_selection,
            identifier=len(self.intensifier_instances),
            seed=self.seed,
            n_seeds=self.n_seeds,
        )
        hp.stats = self.stats
        self.intensifier_instances[len(self.intensifier_instances)] = hp

        return True
