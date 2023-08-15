from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer.random_search import RandomSearch
from smac.facade.abstract_facade import AbstractFacade
from smac.initial_design.default_design import DefaultInitialDesign
from smac.intensifier.intensifier import Intensifier
from smac.model.random_model import RandomModel
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.random_design import AbstractRandomDesign
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class RandomFacade(AbstractFacade):
    """
    Facade to use Random Online Aggressive Racing (ROAR).

    *Aggressive Racing:*
    When we have a new configuration θ, we want to compare it to the current best
    configuration, the incumbent θ*. ROAR uses the 'racing' approach, where we run few times for unpromising θ and many
    times for promising configurations. Once we are confident enough that θ is better than θ*, we update the
    incumbent θ* ⟵ θ. `Aggressive` means rejecting low-performing configurations very early, often after a single run.
    This together is called `aggressive racing`.

    *ROAR Loop:*
    The main ROAR loop looks as follows:

    1. Select a configuration θ uniformly at random.
    2. Compare θ to incumbent θ* online (one θ at a time):
      - Reject/accept θ with `aggressive racing`

    *Setup:*
    Uses a random model and random search for the optimization of the acquisition function.

    Note
    ----
    The surrogate model and the acquisition function is not used during the optimization and therefore replaced
    by dummies.
    """

    @staticmethod
    def get_acquisition_function(scenario: Scenario) -> AbstractAcquisitionFunction:
        """The random facade is not using an acquisition function. Therefore, we simply return a dummy function."""

        class DummyAcquisitionFunction(AbstractAcquisitionFunction):
            def _compute(self, X: np.ndarray) -> np.ndarray:
                return X

        return DummyAcquisitionFunction()

    @staticmethod
    def get_intensifier(
        scenario: Scenario,
        *,
        max_config_calls: int = 3,
        max_incumbents: int = 10,
    ) -> Intensifier:
        """Returns ``Intensifier`` as intensifier.

        Note
        ----
        Please use the ``HyperbandFacade`` if you want to incorporate budgets.

        Warning
        -------
        If you are in an algorithm configuration setting, consider increasing ``max_config_calls``.

        Parameters
        ----------
        max_config_calls : int, defaults to 3
            Maximum number of configuration evaluations. Basically, how many instance-seed keys should be max evaluated
            for a configuration.
        max_incumbents : int, defaults to 10
            How many incumbents to keep track of in the case of multi-objective.
        """
        return Intensifier(
            scenario=scenario,
            max_config_calls=max_config_calls,
            max_incumbents=max_incumbents,
        )

    @staticmethod
    def get_initial_design(
        scenario: Scenario,
        *,
        additional_configs: list[Configuration] = None,
    ) -> DefaultInitialDesign:
        """Returns an initial design, which returns the default configuration.

        Parameters
        ----------
        additional_configs: list[Configuration], defaults to []
            Adds additional configurations to the initial design.
        """
        if additional_configs is None:
            additional_configs = []
        return DefaultInitialDesign(
            scenario=scenario,
            additional_configs=additional_configs,
        )

    @staticmethod
    def get_random_design(scenario: Scenario) -> AbstractRandomDesign:
        """Just like the acquisition function, we do not use a random design. Therefore, we return a dummy design."""

        class DummyRandomDesign(AbstractRandomDesign):
            def check(self, iteration: int) -> bool:
                return True

        return DummyRandomDesign()

    @staticmethod
    def get_model(scenario: Scenario) -> RandomModel:
        """The model is used in the acquisition function. Since we do not use an acquisition function, we return a
        dummy model (returning random values in this case).
        """
        return RandomModel(
            configspace=scenario.configspace,
            instance_features=scenario.instance_features,
            seed=scenario.seed,
        )

    @staticmethod
    def get_acquisition_maximizer(scenario: Scenario) -> RandomSearch:
        """We return ``RandomSearch`` as maximizer which samples configurations randomly from the configuration
        space and therefore neither uses the acquisition function nor the model.
        """
        return RandomSearch(
            scenario.configspace,
            seed=scenario.seed,
        )

    @staticmethod
    def get_multi_objective_algorithm(  # type: ignore
        scenario: Scenario,
        *,
        objective_weights: list[float] | None = None,
    ) -> MeanAggregationStrategy:
        """Returns the mean aggregation strategy for the multi-objective algorithm.

        Parameters
        ----------
        scenario : Scenario
        objective_weights : list[float] | None, defaults to None
            Weights for averaging the objectives in a weighted manner. Must be of the same length as the number of
            objectives.
        """
        return MeanAggregationStrategy(
            scenario=scenario,
            objective_weights=objective_weights,
        )

    @staticmethod
    def get_runhistory_encoder(scenario: Scenario) -> RunHistoryEncoder:
        """Returns the default runhistory encoder."""
        return RunHistoryEncoder(scenario)
