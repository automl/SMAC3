from __future__ import annotations

from smac.acquisition_function.expected_improvement import EI
from smac.acquisition_optimizer.random_search import (
    AbstractAcquisitionOptimizer,
    RandomSearch,
)
from smac.chooser.random_chooser import ChooserProb, RandomChooser
from smac.configspace import Configuration
from smac.facade import Facade
from smac.initial_design import InitialDesign
from smac.initial_design.default_configuration_design import DefaultInitialDesign
from smac.intensification.intensification import Intensifier
from smac.model.random_model import RandomModel
from smac.model.utils import get_types
from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.runhistory.runhistory_transformer import RunhistoryTransformer
from smac.scenario import Scenario

__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class ROAR(Facade):
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

    Following defaults from :class:`~smac.facade.algorithm_configuration.AlgorithmConfigurationFacade` are used:
    - get_acquisition_function
    - get_intensifier
    - get_initial_design
    - get_random_configuration_chooser
    - get_multi_objective_algorithm
    """

    @staticmethod
    def get_acquisition_function(scenario: Scenario, par: float = 0.0) -> EI:
        return EI(par=par)
        # TODO if we use EI, do we still select a configuration with random search?
        #   As far as I understood it, EI won't be used anyway and the acquisition
        #   function optimizer directly samples from the configspace.

    @staticmethod
    def get_intensifier(
        scenario: Scenario,
        *,
        min_challenger: int = 1,
        min_config_calls: int = 1,
        max_config_calls: int = 2000,
    ) -> Intensifier:
        if scenario.deterministic:
            min_challenger = 1

        intensifier = Intensifier(
            instances=scenario.instances,
            instance_specifics=scenario.instance_specifics,  # What is that?
            algorithm_walltime_limit=scenario.algorithm_walltime_limit,
            deterministic=scenario.deterministic,
            min_challenger=min_challenger,
            race_against=scenario.configspace.get_default_configuration(),
            min_config_calls=min_config_calls,
            max_config_calls=max_config_calls,
            seed=scenario.seed,
        )

        return intensifier

    @staticmethod
    def get_initial_design(scenario: Scenario, *, initial_configs: list[Configuration] | None = None) -> InitialDesign:
        return DefaultInitialDesign(
            configspace=scenario.configspace,
            n_runs=scenario.n_runs,
            configs=initial_configs,
            n_configs_per_hyperparameter=0,
            seed=scenario.seed,
        )

    @staticmethod
    def get_random_configuration_chooser(scenario: Scenario, *, random_probability: float = 0.5) -> RandomChooser:
        return ChooserProb(prob=random_probability, seed=scenario.seed)

    @staticmethod
    def get_multi_objective_algorithm(scenario: Scenario) -> AbstractMultiObjectiveAlgorithm | None:
        if len(scenario.objectives) <= 1:
            return None

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
    def get_runhistory_transformer(scenario: Scenario) -> RunhistoryTransformer:
        transformer = RunhistoryTransformer(
            scenario=scenario,
            n_params=len(scenario.configspace.get_hyperparameters()),
            scale_percentage=5,
            seed=scenario.seed,
        )

        return transformer
