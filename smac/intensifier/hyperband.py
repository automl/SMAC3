from __future__ import annotations

from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.intensifier.successive_halving import SuccessiveHalving

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class Hyperband(SuccessiveHalving):
    """Races multiple challengers against an incumbent using Hyperband method.

    Implementation from "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)

    Hyperband is an extension of the Successive Halving intensifier. Please refer to `SuccessiveHalving` documentation
    for more detailed information about the different types of budgets possible and the way instances are handled.

    Internally, this class uses the `HyperbandWorker` class which actually implements the hyperband logic.
    To allow for parallelism, Hyperband can create multiple `HyperbandWorker` instances, based on the number
    of idle workers available.

    Parameters
    ----------
    scenario : Scenario
    instance_seed_pairs : list[tuple[str  |  None, int]] | None, defaults to None
        This argument is used by Hyperband.
    instance_order : str | None, defaults to `shuffle_once`
        How to order the instances. Can be set to:
        * None: Use as is given by the user.
        * shuffle_once: Shuffle once and use across all Successive Halving run .
        * shuffle: Shuffle before every Successive Halving run
    incumbent_selection : str, defaults to `highest_executed_budget`
        How to select the incumbent in Successive Halving. Only active for (real-valued) budgets. Can be set to:
        * highest_executed_budget: Incumbent is the best in the highest budget run so far.
        * highest_budget: Incumbent is selected only based on the highest budget.
        * any_budget: Incumbent is the best on any budget i.e., best performance regardless of budget.
    n_initial_challengers : int | None, defaults to None
        Number of challengers to consider for the initial budget. If not specified, it is calculated internally.
    min_challenger : int, defaults to 1
        Minimal number of challengers to be considered (even if time_bound is exhausted earlier).
    eta : float, defaults to 3
        The "halving" factor after each iteration in a Successive Halving run.
    seed : int | None, defaults to None
    n_seeds : int | None, defaults to None
        The number of seeds to use if the target function is non-deterministic.
    """

    def _get_intensifier_ranking(self, intensifier: AbstractIntensifier) -> tuple[int, int]:
        from smac.intensifier.hyperband_worker import HyperbandWorker

        assert isinstance(intensifier, HyperbandWorker)
        assert intensifier._sh_intensifier

        # For hyperband, we use the internal successive halving as a criteria
        # to see how advanced this intensifier is
        stage = intensifier._sh_intensifier.stage

        return stage, len(intensifier._sh_intensifier._run_tracker)

    def _add_new_instance(self, n_workers: int) -> bool:
        from smac.intensifier.hyperband_worker import HyperbandWorker

        if len(self._intensifier_instances) >= n_workers:
            return False

        hp = HyperbandWorker(
            hyperband=self,
            identifier=len(self._intensifier_instances),
        )
        hp._stats = self._stats
        self._intensifier_instances[len(self._intensifier_instances)] = hp

        return True
