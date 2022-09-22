from __future__ import annotations

from typing import Any
import numpy as np

from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.intensification.parallel_scheduling import ParallelScheduler
from smac.scenario import Scenario
from smac.utils.logging import get_logger
from smac.constants import MAXINT

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class SuccessiveHalving(ParallelScheduler):
    """Races multiple challengers against an incumbent using Successive Halving method.

    Implementation following the description in
    "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)
    Supplementary reference: http://proceedings.mlr.press/v80/falkner18a/falkner18a-supp.pdf

    Successive Halving intensifier (and Hyperband) can operate on two kinds of budgets:

    1. Instances as budget:
      When multiple instances are provided or when run objective is "runtime", this is the criterion used as budget
      for successive halving iterations i.e., the budget determines how many instances the challengers are evaluated
      on at a time. Top challengers for the next iteration are selected based on the combined performance across
      all instances used. If `min_budget` and `max_budget` are not provided, then they are set to 1 and total number
      of available instances respectively by default.
    2. Real-valued budget:
      This is used when there is only one instance provided and when run objective is "quality",
      i.e., budget is a positive, real-valued number that can be passed to the target function as an argument.
      It can be used to control anything by the target function, Eg: number of epochs for training a neural network.
      The parameters `min_budget` and `max_budget` are required for this type of budget.

    This class instantiates ``SuccessiveHalvingWorker`` objects on a need basis, that is, to
    prevent workers from being idle. The actual logic that implements the Successive halving method
    lies in the worker class.

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
    intensify_percentage : float, defaults to 0.5
        How much percentage of the time should configurations be intensified (evaluated on higher budgets or
        more instances). This parameter is accessed in the SMBO class.
    seed : int | None, defaults to None
    n_seeds : int | None, defaults to None
        The number of seeds to use if the target function is non-deterministic.
    """

    def __init__(
        self,
        scenario: Scenario,
        instance_seed_pairs: list[tuple[str | None, int]] | None = None,
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

        self._instance_seed_pairs: list[tuple[str | None, int]]
        self._instance_order = instance_order
        self._incumbent_selection = incumbent_selection
        self._n_initial_challengers = n_initial_challengers
        self._n_seeds = n_seeds if n_seeds else 1
        self._eta = eta
        self._min_budget: float | int
        self._max_budget: float | int
        self._instance_as_budget = False

        available_incumbent_selections = ["highest_executed_budget", "highest_budget", "any_budget"]
        if incumbent_selection not in available_incumbent_selections:
            raise ValueError(f"The incumbent selection must be one of {available_incumbent_selections}.")

        if self._min_challenger > 1:
            raise ValueError("Successive Halving can not handle argument `min_challenger` > 1.")

        if self._instances is not None and len(self._instances) == 1 and self._n_seeds > 1:
            raise NotImplementedError("The case multiple seeds and one instance can not be handled yet!")

        if eta <= 1:
            raise ValueError("The parameter `eta` must be greater than 1.")

        # We declare the instance seed pairs now
        if instance_seed_pairs is None:
            # Set seed(s) for all SH runs. Currently user gives the number of seeds to consider.
            if self._deterministic:
                seeds = [0]
                logger.info("Only one seed is used as `deterministic` is set to true.")
            else:
                seeds = [int(s) for s in self._rng.randint(low=0, high=MAXINT, size=self._n_seeds)]
                if self._n_seeds == 1:
                    logger.warning(
                        "The target function is specified to be non-deterministic, "
                        "but number of seeds to evaluate are set to 1. "
                        "Consider increasing `n_seeds` from the intensifier."
                    )

            # Storing instances & seeds as tuples
            if self._instances is not None:
                self._instance_seed_pairs = [(i, s) for s in seeds for i in self._instances]
            else:
                self._instance_seed_pairs = [(None, s) for s in seeds]

            # Determine instance-seed pair order
            if self._instance_order == "shuffle_once":
                # Randomize once
                self._rng.shuffle(self._instance_seed_pairs)  # type: ignore
        else:
            self._instance_seed_pairs = instance_seed_pairs

        # Budgets
        min_budget = scenario.min_budget
        max_budget = scenario.max_budget

        if max_budget is not None and min_budget is not None and max_budget < min_budget:
            raise ValueError("Max budget has to be larger than min budget.")

        # If only 1 instance was provided & quality objective, then use algorithm_walltime_limit as budget
        # else, use instances as budget
        if self._instance_seed_pairs is None or len(self._instance_seed_pairs) <= 1:
            # budget with algorithm_walltime_limit
            if min_budget is None or max_budget is None:
                raise ValueError(
                    "Successive Halving with real-valued budget (i.e., only 1 instance) "
                    "requires parameters `min_budget` and `max_budget` for intensification!"
                )

            self._min_budget = min_budget
            self._max_budget = max_budget
            self._instance_as_budget = False
        else:
            # Budget with instances
            self._min_budget = int(min_budget) if min_budget else 1
            self._max_budget = int(max_budget) if max_budget else len(self._instance_seed_pairs)
            self._instance_as_budget = True

            if self._max_budget > len(self._instance_seed_pairs):
                raise ValueError("Max budget can not be greater than the number of instance-seed pairs.")
            if self._max_budget < len(self._instance_seed_pairs):
                logger.warning(
                    "Max budget (%d) does not include all instance-seed pairs (%d)."
                    % (self._max_budget, len(self._instance_seed_pairs))
                )

        budget_type = "INSTANCES" if self._instance_as_budget else "BUDGETS"
        logger.info(
            f"Using successive halving with budget type {budget_type}, min budget {self._min_budget}, "
            f"max budget {self._max_budget} and eta {self._eta}."
        )

        # Pre-computing stuff for SH
        self._max_sh_iterations = int(np.floor(np.log(self._max_budget / self._min_budget) / np.log(self._eta)))

        # Initial number of challengers to sample
        if n_initial_challengers is None:
            self._n_initial_challengers = int(self._eta**self._max_sh_iterations)

        # Challengers can be repeated only if optimizing across multiple seeds or changing instance orders every run
        # (this does not include having multiple instances)
        if self._n_seeds > 1 or self._instance_order == "shuffle":
            self._repeat_configs = True
        else:
            self._repeat_configs = False

        if self._instance_as_budget:
            logger.info("The argument `incumbent_selection` is ignored because instances are used as budget type.")

    def get_meta(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "instance_seed_pairs": self._instance_seed_pairs,
            "instance_order": self._instance_order,
            "incumbent_selection": self._incumbent_selection,
            "n_initial_challengers": self._n_initial_challengers,
            "min_challenger": self._min_challenger,
            "eta": self._eta,
            "intensify_percentage": self._intensify_percentage,
            "seed": self._seed,
            "n_seeds": self._n_seeds,
        }

    @property
    def uses_budgets(self) -> bool:
        return not self._instance_as_budget

    @property
    def uses_instances(self) -> bool:
        return self._instance_as_budget

    def _get_intensifier_ranking(self, intensifier: AbstractIntensifier) -> tuple[int, int]:
        from smac.intensification.successive_halving_worker import SuccessiveHalvingWorker

        assert isinstance(intensifier, SuccessiveHalvingWorker)

        # Each row of this matrix is id, stage, configs+instances for stage
        # We use sh.run_tracker as a cheap way to know how advanced the run is
        # in case of stage ties among successive halvers. sh.run_tracker is
        # also emptied each iteration
        return intensifier.stage, len(intensifier._run_tracker)

    def _add_new_instance(self, n_workers: int) -> bool:
        from smac.intensification.successive_halving_worker import SuccessiveHalvingWorker

        if len(self._intensifier_instances) >= n_workers:
            return False

        sh = SuccessiveHalvingWorker(
            successive_halving=self,
            identifier=len(self._intensifier_instances),
        )
        sh._stats = self._stats
        self._intensifier_instances[len(self._intensifier_instances)] = sh

        return True
