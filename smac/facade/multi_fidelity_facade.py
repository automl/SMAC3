from __future__ import annotations

from ConfigSpace import Configuration
from smac.facade.hyperparameter_facade import HyperparameterFacade
from smac.initial_design.random_design import RandomInitialDesign
from smac.intensification.hyperband import Hyperband
from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class MultiFidelityFacade(HyperparameterFacade):
    """
    This facade configures SMAC in a multi-fidelity setting.
    The way this facade combines the components is the following and exploits
    fidelity information in the following form:

    1. The initial design is a RandomInitialDesign.
    2. The intensification is Hyperband. The configurations from the initial design
    are presented as challengers and executed in the Hyperband fashion.
    3. The model is a RandomForest surrogate model. The data to train it
    is collected by SMBO._collect_data. Notably, the method searches through
    the runhistory and collects the data from the highest fidelity level, that supports at least
    SMBO._min_samples_model number of configurations.
    4. The acquisition function is EI. Presenting the value of a candidate configuration.
    5. The acquisition optimizer is LocalAndSortedRandomSearch. It optimizes the acquisition
    function to present the best configuration as challenger to the intensifier.
    From now on 2. works as follows: the intensifier runs the challenger in a Hyperband fashion
    against the existing configurations and their observed performances
    until the challenger does not survive a fidelity level. The intensifier can inquire about a
    known configuration on a yet unseen fidelity if necessary.

    The loop 2-5 continues until a termination criterion is reached.

    :Note:
    For intensification the data acquisition and aggregation strategy in step 2 is changed.
    Incumbents are updated by the mean performance over the intersection of instances, that
    the challenger and incumbent have in common (abstract_intensifier._compare_configs).
    The model in step 3 is trained on the all the available instance performance values.
    The datapoints for a hyperparameter configuration are disambiguated by the instance features
    or an index as replacement if no instance features are available.
    """

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        eta: int = 3,
        min_challenger: int = 1,
        intensify_percentage: float = 0.5,
        n_seeds: int = 1,
    ) -> Hyperband:
        """Returns a Hyperband intensifier instance. Please check its documentation for details."""
        return Hyperband(
            scenario=scenario,
            eta=eta,
            min_challenger=min_challenger,
            intensify_percentage=intensify_percentage,
            n_seeds=n_seeds,
        )

    @staticmethod
    def get_initial_design(  # type: ignore
        scenario: Scenario,
        *,
        configs: list[Configuration] | None = None,
        n_configs: int | None = None,
        n_configs_per_hyperparamter: int = 10,
        max_config_ratio: float = 0.25,  # Use at most X*budget in the initial design
    ) -> RandomInitialDesign:
        """Returns a random initial design instance. Please check its documentation for details."""
        return RandomInitialDesign(
            scenario=scenario,
            configs=configs,
            n_configs=n_configs,
            n_configs_per_hyperparameter=n_configs_per_hyperparamter,
            max_config_ratio=max_config_ratio,
        )
