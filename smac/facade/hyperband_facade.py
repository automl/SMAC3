from __future__ import annotations


from smac.facade.random_facade import RandomFacade
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class HyperbandFacade(RandomFacade):
    """
    Facade to use model-free Hyperband [1]_ for algorithm configuration.

    Use ROAR (Random Aggressive Online Racing) to compare configurations, a random
    initial design and the Hyperband intensifier.
    """

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        eta: int = 3,
        min_challenger: int = 1,
    ) -> Hyperband:
        """Returns a Hyperband intensifier instance. That means that budgets are supported."""
        return Hyperband(
            scenario=scenario,
            eta=eta,
            min_challenger=min_challenger,
        )
