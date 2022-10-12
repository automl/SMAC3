from __future__ import annotations

from ConfigSpace import Configuration

from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade,
)
from smac.initial_design.random_design import RandomInitialDesign
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class MultiFidelityFacade(HyperparameterOptimizationFacade):
    """
    This facade configures SMAC in a multi-fidelity setting.
    The way this facade combines the components is the following and exploits
    fidelity information in the following form:

    1. The initial design is a RandomInitialDesign.
    2. The intensification is Hyperband. The configurations from the initial design
    are presented as challengers and executed in the Hyperband fashion.
    3. The model is a RandomForest surrogate model. The data to train it is collected by ``SMBO._collect_data``.
      Notably, the method searches through the runhistory and collects the data from the highest fidelity level,
      that supports at least ``SMBO._min_samples_model`` number of configurations.
    4. The acquisition function is ``EI``, presenting the value of a candidate configuration.
    5. The acquisition optimizer is ``LocalAndSortedRandomSearch``. It optimizes the acquisition
      function to present the best configuration as challenger to the intensifier.
      From now on 2. works as follows: The intensifier runs the challenger in a Hyperband fashion against the existing
      configurations and their observed performances until the challenger does not survive a fidelity level. The
      intensifier can inquire about a known configuration on a yet unseen fidelity if necessary.

    The loop 2-5 continues until a termination criterion is reached.

    Note
    ----
    For intensification the data acquisition and aggregation strategy in step 2 is changed.
    Incumbents are updated by the mean performance over the intersection of instances, that
    the challenger and incumbent have in common (``abstract_intensifier._compare_configs``).
    The model in step 3 is trained on all the available instance performance values.
    The datapoints for a hyperparameter configuration are disambiguated by the instance features
    or an index as replacement if no instance features are available.
    """

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        eta: int = 3,
        min_challenger: int = 1,
        n_seeds: int = 1,
    ) -> Hyperband:
        """Returns a Hyperband intensifier instance. That means that budgets are supported.

        min_challenger : int, defaults to 1
            Minimal number of challengers to be considered (even if time_bound is exhausted earlier).
        eta : float, defaults to 3
            The "halving" factor after each iteration in a Successive Halving run.
        n_seeds : int | None, defaults to None
            The number of seeds to use if the target function is non-deterministic.
        """
        return Hyperband(
            scenario=scenario,
            eta=eta,
            min_challenger=min_challenger,
            n_seeds=n_seeds,
        )

    @staticmethod
    def get_initial_design(  # type: ignore
        scenario: Scenario,
        *,
        n_configs: int | None = None,
        n_configs_per_hyperparamter: int = 10,
        max_ratio: float = 0.1,
        additional_configs: list[Configuration] = [],
    ) -> RandomInitialDesign:
        """Returns a random initial design.

        Parameters
        ----------
        scenario : Scenario
        n_configs : int | None, defaults to None
            Number of initial configurations (disables the arguments ``n_configs_per_hyperparameter``).
        n_configs_per_hyperparameter: int, defaults to 10
            Number of initial configurations per hyperparameter. For example, if my configuration space covers five
            hyperparameters and ``n_configs_per_hyperparameter`` is set to 10, then 50 initial configurations will be
            samples.
        max_ratio: float, defaults to 0.1
            Use at most ``scenario.n_trials`` * ``max_ratio`` number of configurations in the initial design.
            Additional configurations are not affected by this parameter.
        additional_configs: list[Configuration], defaults to []
            Adds additional configurations to the initial design.
        """
        return RandomInitialDesign(
            scenario=scenario,
            n_configs=n_configs,
            n_configs_per_hyperparameter=n_configs_per_hyperparamter,
            max_ratio=max_ratio,
            additional_configs=additional_configs,
        )
