from __future__ import annotations

from typing import TypeVar

import numpy as np

from smac.model.abstract_model import AbstractModel

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


Self = TypeVar("Self", bound="MultiObjectiveModel")


class MultiObjectiveModel(AbstractModel):
    """Wrapper for the surrogate model to predict multiple objectives.

    Parameters
    ----------
    models : AbstractModel | list[AbstractModel]
        Which model should be used. If it is a list, then it must provide as many models as objectives.
        If it is a single model only, the model is used for all objectives.
    objectives : list[str]
        Which objectives should be used.
    seed : int
    """

    def __init__(
        self,
        models: AbstractModel | list[AbstractModel],
        objectives: list[str],
        seed: int = 0,
    ) -> None:
        self._n_objectives = len(objectives)
        if isinstance(models, list):
            assert len(models) == len(objectives)

            # Make sure the configspace is the same
            configspace = models[0]._configspace
            for m in models:
                assert configspace == m._configspace

            self._models = models
        else:
            configspace = models._configspace
            self._models = [models for _ in range(self._n_objectives)]

        super().__init__(
            configspace=configspace,
            instance_features=None,
            pca_components=None,
            seed=seed,
        )

    @property
    def models(self) -> list[AbstractModel]:
        """The internally used surrogate models."""
        return self._models

    def predict_marginalized(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: D102
        mean = np.zeros((X.shape[0], self._n_objectives))
        var = np.zeros((X.shape[0], self._n_objectives))

        for i, estimator in enumerate(self._models):
            m, v = estimator.predict_marginalized(X)
            mean[:, i] = m.flatten()
            var[:, i] = v.flatten()

        return mean, var

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
