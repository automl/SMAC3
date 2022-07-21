from smac.acquisition_optimizer.random_search import RandomSearch, AbstractAcquisitionOptimizer
from smac.scenario import Scenario
from smac.facade.algorithm_configuration import AlgorithmConfigurationFacade
from smac.model.random_model import RandomModel
from smac.runhistory.runhistory_transformer import RunhistoryTransformer
from smac.model.utils import get_types

__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class ROAR(AlgorithmConfigurationFacade):
    """
    Facade to use ROAR mode.
    """
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
