from __future__ import annotations

from typing import Any


from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.intensification.parallel_scheduling import ParallelScheduler
from smac.scenario import Scenario
from smac.intensification.successive_halving_worker import SuccessiveHalvingWorker
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class SuccessiveHalving(ParallelScheduler):
    """Races multiple challengers against an incumbent using Successive Halving method.

    Implementation following the description in
    "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)
    Supplementary reference: http://proceedings.mlr.press/v80/falkner18a/falkner18a-supp.pdf

    Successive Halving intensifier (and Hyperband) can operate on two kinds of budgets:
    1. **'Instances' as budget**:
        When multiple instances are provided or when run objective is "runtime", this is the criterion used as budget
        for successive halving iterations i.e., the budget determines how many instances the challengers are evaluated
        on at a time. Top challengers for the next iteration are selected based on the combined performance across
        all instances used.

        If ``min_budget`` and ``max_budget`` are not provided, then they are set to 1 and total number
        of available instances respectively by default.

    2. **'Real-valued' budget**:
        This is used when there is only one instance provided and when run objective is "quality",
        i.e., budget is a positive, real-valued number that can be passed to the target algorithm as an argument.
        It can be used to control anything by the target algorithm, Eg: number of epochs for training a neural network.

        ``min_budget`` and ``max_budget`` are required parameters for this type of budget.

    Examples for successive halving (and hyperband) can be found here:
    * Runtime objective and multiple instances *(instances as budget)*: `examples/spear_qcp/SMAC4AC_SH_spear_qcp.py`
    * Quality objective and multiple instances *(instances as budget)*: `examples/SMAC4MF_sgd_instances.py`
    * Quality objective and single instance *(real-valued budget)*: `examples/SMAC4MF_mlp.py`

    This class instantiates `_SuccessiveHalving` objects on a need basis, that is, to
    prevent workers from being idle. The actual logic that implements the Successive halving method
    lies on the _SuccessiveHalving class.

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
        algorithm_walltime_limit of TA runs
    deterministic : bool
        whether the TA is deterministic or not
    min_budget : Optional[float]
        minimum budget allowed for 1 run of successive halving
    max_budget : Optional[float]
        maximum budget allowed for 1 run of successive halving
    eta : float
        'halving' factor after each iteration in a successive halving run. Defaults to 3
    n_initial_challengers : Optional[int]
        number of challengers to consider for the initial budget. If None, calculated internally
    n_seeds : Optional[int]
        Number of seeds to use, if TA is not deterministic. Defaults to None, i.e., seed is set as 0
    instance_order : Optional[str]
        how to order instances. Can be set to: [None, shuffle_once, shuffle]
        * None - use as is given by the user
        * shuffle_once - shuffle once and use across all SH run (default)
        * shuffle - shuffle before every SH run
    instance_seed_pairs : List[Tuple[str, int]], optional
        Do not set this argument, it will only be used by hyperband!
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
        n_initial_challengers: int | None = None,
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
        self.n_initial_challengers = n_initial_challengers
        self.eta = eta
        self.n_seeds = n_seeds

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def _get_intensifier_ranking(self, intensifier: AbstractIntensifier) -> tuple[int, int]:
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
        assert isinstance(intensifier, SuccessiveHalvingWorker)

        # Each row of this matrix is id, stage, configs+instances for stage
        # We use sh.run_tracker as a cheap way to know how advanced the run is
        # in case of stage ties among successive halvers. sh.run_tracker is
        # also emptied each iteration
        stage = 0
        if hasattr(intensifier, "stage"):
            # Newly created SuccessiveHalvingWorker objects have no stage
            stage = intensifier.stage

        return stage, len(intensifier.run_tracker)

    def _add_new_instance(self, n_workers: int) -> bool:
        """Decides if it is possible to add a new intensifier instance, and adds it. If a new
        intensifier instance is added, True is returned, else False.

        Parameters
        ----------
        n_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
        Whether or not a successive halving instance was added
        """
        assert self.stats

        if len(self.intensifier_instances) >= n_workers:
            return False

        sh = SuccessiveHalvingWorker(
            scenario=self.scenario,
            instance_seed_pairs=self.instance_seed_pairs,
            instance_order=self.instance_order,
            incumbent_selection=self.incumbent_selection,
            n_initial_challengers=self.n_initial_challengers,
            min_challenger=self.min_challenger,
            eta=self.eta,
            intensify_percentage=self.intensify_percentage,
            seed=self.seed,
            n_seeds=self.n_seeds,
            identifier=len(self.intensifier_instances),
        )
        sh.stats = self.stats
        self.intensifier_instances[len(self.intensifier_instances)] = sh

        return True
