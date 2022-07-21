from __future__ import annotations

from smac.configspace import Configuration
from smac.facade.random import ROAR
from smac.initial_design import InitialDesign
from smac.initial_design.random_configuration_design import RandomInitialDesign
from smac.intensification.hyperband import Hyperband
from smac.scenario import Scenario

__author__ = "Ashwin Raaghav Narayanan"
__copyright__ = "Copyright 2019, ML4AAD"
__license__ = "3-clause BSD"


class HB4AC(ROAR):
    """
    Facade to use model-free Hyperband [1]_ for algorithm configuration.

    Use ROAR (Random Aggressive Online Racing) to compare configurations, a random
    initial design and the Hyperband intensifier.

    .. [1] Lisha Li, Kevin G. Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, Ameet Talwalkar:
        Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.
        J. Mach. Learn. Res. 18: 185:1-185:52 (2017)
        https://jmlr.org/papers/v18/16-558.html

    """

    @staticmethod
    def get_initial_design(scenario: Scenario, *, initial_configs: list[Configuration] | None = None) -> InitialDesign:
        return RandomInitialDesign(
            configspace=scenario.configspace,
            n_runs=scenario.n_runs,
            configs=initial_configs,
            n_configs_per_hyperparameter=0,
            seed=scenario.seed,
        )

    @staticmethod
    def get_intensifier(
        scenario: Scenario,
        *,
        min_challenger: int = 1,
        instance_order: str = "shuffle_once",
    ) -> Hyperband:
        return Hyperband(
            instances=scenario.instances,
            instance_specifics=scenario.instance_specifics,
            algorithm_walltime_limit=scenario.algorithm_walltime_limit,
            deterministic=scenario.deterministic,
            initial_budget=scenario.initial_budget,
            max_budget=scenario.max_budget,
            eta=scenario.eta,
            min_challenger=min_challenger,
            seed=scenario.seed,
        )
