from __future__ import annotations

from smac.facade.random_facade import RandomFacade
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class HyperbandFacade(RandomFacade):
    """
    Facade to use model-free Hyperband [LJDR18]_ for algorithm configuration.

    Uses Random Aggressive Online Racing (ROAR) to compare configurations, a random
    initial design and the Hyperband intensifier.
    """

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        min_challenger: int = 1,
        eta: int = 3,
    ) -> Hyperband:
        """Returns a Hyperband intensifier instance. That means that budgets are supported.

        min_challenger : int, defaults to 1
            Minimal number of challengers to be considered (even if time_bound is exhausted earlier).
        eta : float, defaults to 3
            The "halving" factor after each iteration in a Successive Halving run.
        """
        return Hyperband(
            scenario=scenario,
            min_challenger=min_challenger,
            eta=eta,
        )
