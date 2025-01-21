from __future__ import annotations

from smac.facade.random_facade import RandomFacade
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class HyperbandFacade(RandomFacade):
    """
    Facade to use model-free Hyperband [[LJDR18][LJDR18]] for algorithm configuration.

    Uses Random Aggressive Online Racing (ROAR) to compare configurations, a random
    initial design and the Hyperband intensifier.


    !!! warning
        ``smac.main.config_selector.ConfigSelector`` contains the ``min_trials`` parameter. This parameter determines
        how many samples are required to train the surrogate model. If budgets are involved, the highest budgets
        are checked first. For example, if min_trials is three, but we find only two trials in the runhistory for
        the highest budget, we will use trials of a lower budget instead.
    """

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        eta: int = 3,
        n_seeds: int = 1,
        instance_seed_order: str | None = "shuffle_once",
        max_incumbents: int = 10,
        incumbent_selection: str = "highest_observed_budget",
    ) -> Hyperband:
        """Returns a Hyperband intensifier instance. Budgets are supported.

        eta : int, defaults to 3
            Input that controls the proportion of configurations discarded in each round of Successive Halving.
        n_seeds : int, defaults to 1
            How many seeds to use for each instance.
        instance_seed_order : str, defaults to "shuffle_once"
            How to order the instance-seed pairs. Can be set to:
            * None: No shuffling at all and use the instance-seed order provided by the user.
            * "shuffle_once": Shuffle the instance-seed keys once and use the same order across all runs.
            * "shuffle": Shuffle the instance-seed keys for each bracket individually.
        incumbent_selection : str, defaults to "any_budget"
            How to select the incumbent when using budgets. Can be set to:
            * "any_budget": Incumbent is the best on any budget i.e., best performance regardless of budget.
            * "highest_observed_budget": Incumbent is the best in the highest budget run so far.
            * "highest_budget": Incumbent is selected only based on the highest budget.
        max_incumbents : int, defaults to 10
            How many incumbents to keep track of in the case of multi-objective.
        """
        return Hyperband(
            scenario=scenario,
            eta=eta,
            n_seeds=n_seeds,
            instance_seed_order=instance_seed_order,
            max_incumbents=max_incumbents,
            incumbent_selection=incumbent_selection,
        )
