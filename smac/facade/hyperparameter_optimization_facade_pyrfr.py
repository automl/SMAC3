from __future__ import annotations

from smac.scenario import Scenario
from smac.model.random_forest.random_forest_old import RandomForest as PyrfrForest
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade


class HyperparameterOptimizationRFRFacade(HyperparameterOptimizationFacade):
    @staticmethod
    def get_model(  # type: ignore
            scenario: Scenario,
            *,
            n_trees: int = 10,
            ratio_features: float = 1.0,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            max_depth: int = 2 ** 20,
            bootstrapping: bool = True,
    ) -> PyrfrForest:
        """Returns a random forest as surrogate model.

        Parameters
        ----------
        n_trees : int, defaults to 10
            The number of trees in the random forest.
        ratio_features : float, defaults to 5.0 / 6.0
            The ratio of features that are considered for splitting.
        min_samples_split : int, defaults to 3
            The minimum number of data points to perform a split.
        min_samples_leaf : int, defaults to 3
            The minimum number of data points in a leaf.
        max_depth : int, defaults to 20
            The maximum depth of a single tree.
        bootstrapping : bool, defaults to True
            Enables bootstrapping.
        """
        return PyrfrForest(
            log_y=True,
            n_trees=n_trees,
            bootstrapping=bootstrapping,
            ratio_features=ratio_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            configspace=scenario.configspace,
            instance_features=scenario.instance_features,
            seed=scenario.seed,
        )