from __future__ import annotations

from ConfigSpace import Configuration

from smac.facade.random_facade import RandomFacade
from smac.initial_design.random_design import RandomInitialDesign
from smac.intensification.hyperband import Hyperband
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
        intensify_percentage: float = 0.5,
    ) -> Hyperband:
        """Returns a Hyperband intensifier instance. That means that budgets are supported."""
        return Hyperband(
            scenario=scenario,
            eta=eta,
            min_challenger=min_challenger,
            intensify_percentage=intensify_percentage,
        )
