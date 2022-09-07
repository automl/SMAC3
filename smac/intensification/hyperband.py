from __future__ import annotations

from typing import Any, Tuple

from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.intensification.successive_halving import SuccessiveHalving

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class Hyperband(SuccessiveHalving):
    """Races multiple challengers against an incumbent using Hyperband method.

    Implementation from "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)

    Hyperband is an extension of the Successive Halving intensifier. Please refer to `SuccessiveHalving` documentation
    for more detailed information about the different types of budgets possible and the way instances are handled.

    Internally, this class uses the _Hyperband private class which actually implements the hyperband logic.
    To allow for parallelism, Hyperband can create multiple _Hyperband instances, based on the number
    of idle workers available.

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
    """

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
        from smac.intensification.hyperband_worker import HyperbandWorker

        assert isinstance(intensifier, HyperbandWorker)
        assert intensifier._sh_intensifier

        # For hyperband, we use the internal successive halving as a criteria
        # to see how advanced this intensifier is
        stage = intensifier._sh_intensifier.stage

        return stage, len(intensifier._sh_intensifier._run_tracker)

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
        from smac.intensification.hyperband_worker import HyperbandWorker

        if len(self._intensifier_instances) >= n_workers:
            return False

        hp = HyperbandWorker(
            hyperband=self,
            identifier=len(self._intensifier_instances),
        )
        hp._stats = self._stats
        self._intensifier_instances[len(self._intensifier_instances)] = hp

        return True
