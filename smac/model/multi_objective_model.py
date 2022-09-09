from __future__ import annotations

from typing import TypeVar

import numpy as np

from ConfigSpace import ConfigurationSpace
from smac.model.abstract_model import AbstractModel

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


Self = TypeVar("Self", bound="MultiObjectiveModel")


class MultiObjectiveModel(AbstractModel):
    """Wrapper for the surrogate model to predict multiple objectives.

    Parameters
    ----------
    configspace : ConfigurationSpace
    model : AbstractModel
        Which model should be used for each objective.
    objectives : list[str]
        Which objectives should be used.
    instance_features : dict[str, list[int | float]] | None, defaults to None
        Features (list of int or floats) of the instances (str). The features are incorporated into the X data,
        on which the model is trained on.
    pca_components : float, defaults to 7
        Number of components to keep when using PCA to reduce dimensionality of instance features.
    seed : int
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        model: AbstractModel,
        objectives: list[str],
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
    ) -> None:
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )

        self._n_objectives = len(objectives)
        self._models: list[AbstractModel] = [model for _ in range(self._n_objectives)]

    def _train(self: Self, X: np.ndarray, Y: np.ndarray) -> Self:
        if len(self._models) == 0:
            raise ValueError("The list of surrogate models is empty.")

        for i, model in enumerate(self._models):
            model.train(X, Y[:, i])

        return self

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if covariance_type != "diagonal":
            raise ValueError("`covariance_type` can only take `diagonal` for this model.")

        mean = np.zeros((X.shape[0], self._n_objectives))
        var = np.zeros((X.shape[0], self._n_objectives))

        for i, estimator in enumerate(self._models):
            m, v = estimator.predict(X)
            assert v is not None
            mean[:, i] = m.flatten()
            var[:, i] = v.flatten()

        return mean, var

    def predict_marginalized(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros((X.shape[0], self._n_objectives))
        var = np.zeros((X.shape[0], self._n_objectives))

        for i, estimator in enumerate(self._models):
            m, v = estimator.predict_marginalized(X)
            mean[:, i] = m.flatten()
            var[:, i] = v.flatten()

        return mean, var
