from __future__ import annotations

from smac.acquisition.functions.expected_improvement import EI
from smac.acquisition.random_search import AbstractAcquisitionOptimizer, RandomSearch
from smac.random_design import RandomDesign, ProbabilityRandomDesign
from smac.configspace import Configuration
from smac.facade.facade import Facade
from smac.initial_design.default_design import DefaultInitialDesign
from smac.intensification.intensification import Intensifier
from smac.model.random_model import RandomModel
from smac.model.utils import get_types
from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class RandomFacade(Facade):
    """
    Facade to use ROAR (Random Online Aggressive Racing) mode.

    Aggressive Racing
    -----------------
    When we have a new configuration θ we want to compare it to the current best
    configuration, the incumbent θ*.
    ROAR uses the 'racing' approach, where we run few times for unpromising θ and many
    times for promising configurations. Once we are confident enough that θ is better
    than θ*, we update the incumbent θ* ⟵ θ.
    'Aggressive' means rejecting low-performing configurations very early, often after
    a single run.
    This together is called 'aggressive racing'.

    ROAR Loop
    ---------
    The main ROAR loop looks as follows:
        1. Select a configuration θ uniformly at random.
        2. Compare θ to incumbent θ* online (one θ at a time):
            - Reject/accept θ with 'aggressive racing'.

    ROAR Setup
    ----------
    Use a random model and random search for the optimization of the acquisition function.

    Following defaults from :class:
    `~smac.facade.algorithm_configuration.AlgorithmConfigurationFacade` are used:
    - get_acquisition_function
    - get_intensifier
    - get_initial_design
    - get_random_design
    - get_multi_objective_algorithm
    """

    @staticmethod
    def get_acquisition_function(scenario: Scenario, xi: float = 0.0) -> EI:
        # TODO: if we use EI, do we still select a configuration with random search?
        # As far as I understood it, EI won't be used anyway and the acquisition
        # function optimizer directly samples from the configspace.

        return EI(xi=xi)

    @staticmethod
    def get_intensifier(
        scenario: Scenario,
        *,
        min_challenger: int = 1,
        min_config_calls: int = 1,
        max_config_calls: int = 2000,
        intensify_percentage: float = 0.5,
    ) -> Intensifier:
        intensifier = Intensifier(
            scenario=scenario,
            min_challenger=min_challenger,
            race_against=scenario.configspace.get_default_configuration(),
            min_config_calls=min_config_calls,
            max_config_calls=max_config_calls,
            intensify_percentage=intensify_percentage,
        )

        return intensifier

    @staticmethod
    def get_initial_design(
        scenario: Scenario,
        *,
        configs: list[Configuration] | None = None,
    ) -> DefaultInitialDesign:
        return DefaultInitialDesign(
            scenario=scenario,
            configs=configs,
        )

    @staticmethod
    def get_random_design(
        scenario: Scenario,
        *,
        random_probability: float = 0.5,
    ) -> RandomDesign:
        return ProbabilityRandomDesign(probability=random_probability, seed=scenario.seed)

    @staticmethod
    def get_multi_objective_algorithm(scenario: Scenario) -> AbstractMultiObjectiveAlgorithm:
        return MeanAggregationStrategy(scenario.seed)

    @staticmethod
    def get_model(
        scenario: Scenario,
        *,
        pca_components: int = 4,
    ) -> RandomModel:
        types, bounds = get_types(scenario.configspace, scenario.instance_features)

        return RandomModel(
            configspace=scenario.configspace,
            types=types,
            bounds=bounds,
            seed=scenario.seed,
            instance_features=scenario.instance_features,
            pca_components=pca_components,
        )

    @staticmethod
    def get_acquisition_optimizer(scenario: Scenario) -> AbstractAcquisitionOptimizer:
        optimizer = RandomSearch(
            scenario.configspace,
            seed=scenario.seed,
        )

        return optimizer

    @staticmethod
    def get_runhistory_encoder(scenario: Scenario) -> RunHistoryEncoder:
        return RunHistoryEncoder(scenario)
