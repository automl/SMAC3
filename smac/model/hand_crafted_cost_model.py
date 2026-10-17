from __future__ import annotations

from typing import Callable

import numpy as np
from ConfigSpace import Configuration

from smac.model.abstract_model import AbstractModel
from smac.model.surrogate_transformer import SurrogateTransformer
from smac.scenario import Scenario
from smac.utils.logging import get_logger


class HandCraftedCostModel(AbstractModel):
    """A deterministic cost model that evaluates an explicit cost formula.

    When evaluation costs are known a priori from configuration hyperparameters
    (such as the number of epochs or dataset subsample size) or from an analytical
    cost formula, fitting an empirical surrogate model to observed trial costs is
    unnecessary and prone to approximation errors.

    ``HandCraftedCostModel`` directly evaluates the provided ``cost_formula`` on
    input configurations. Its :meth:`train` method is a no-op, and :meth:`predict`
    evaluates the formula with zero variance.
    """

    def __init__(
        self,
        scenario: Scenario,
        cost_formula: Callable[[Configuration], float],
    ):
        super().__init__(configspace=scenario.configspace, seed=scenario.seed)
        self._cost_formula = cost_formula
        self._logger = get_logger(self.__class__.__name__)
        self.transformer = self.build_transformer()

    def build_transformer(self, normalize_y: bool = False) -> SurrogateTransformer:
        """Returns a pass-through transformer.

        No imputation, scaling, PCA or y-normalization is needed because this
        model applies a fixed formula directly to the raw configuration vectors.
        """
        return SurrogateTransformer(
            n_hps=self._n_hps,
            n_features=self._n_features,
            instance_features=self._instance_features,
            impute_inactive=None,
            normalize_y=normalize_y,
            pca_components=None,
        )

    def _train(self, X: np.ndarray, y: np.ndarray) -> HandCraftedCostModel:
        """The model is formulaic, so training is a no-op."""
        self._logger.info("HandCraftedCostModel does not learn from data. Skipping training.")
        return self

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predicts costs by applying the formula.

        Variance is always zero as the model is deterministic.
        """
        costs = np.array([self._cost_formula(Configuration(self._configspace, vector=x[: self._n_hps])) for x in X])
        variances = np.zeros_like(costs)
        return costs, variances
