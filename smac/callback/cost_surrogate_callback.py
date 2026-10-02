from __future__ import annotations

import numpy as np

from smac.callback.callback import Callback
from smac.main.smbo import SMBO
from smac.model.abstract_model import AbstractModel
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runhistory.encoder.abstract_encoder import AbstractRunHistoryEncoder
from smac.runhistory.encoder.log_encoder import RunHistoryLogEncoder
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class CostSurrogateCallback(Callback):
    """Callback that maintains a cost surrogate model trained on observed evaluation costs.

    A separate ``RunHistory`` stores the resource cost of each trial in the ``cost``
    field so that the standard encoder pipeline can be reused to build ``(X, Y)``
    matrices for training.

    Parameters
    ----------
    cost_model : AbstractModel
        The surrogate model to be trained on the cost data.
    scenario : Scenario
        The SMAC scenario object.
    cost_encoder : AbstractRunHistoryEncoder | None, defaults to None
        The encoder to transform the cost data before training the model.
        If None, defaults to ``RunHistoryLogEncoder`` (plain ``log`` transform).
    """

    _cost_encoder: AbstractRunHistoryEncoder

    def __init__(
        self,
        cost_model: AbstractModel,
        scenario: Scenario,
        cost_encoder: AbstractRunHistoryEncoder | None = None,
    ):
        self._scenario = scenario
        self._cost_model = cost_model
        self._cost_runhistory = RunHistory()

        if cost_encoder is None:
            self._cost_encoder = RunHistoryLogEncoder(scenario=scenario)
        else:
            self._cost_encoder = cost_encoder

        self._cost_encoder.runhistory = self._cost_runhistory

        # Wire the encoder's response transform (e.g. log) into the model's
        # training pipeline so that `model.train(X, Y_raw)` automatically
        # applies the transform before fitting.  Predictions then come back
        # in the transformed space (e.g. log-space), which `predict_cost`
        # inverts below.
        if not isinstance(self._cost_model, HandCraftedCostModel):
            self._cost_model.set_y_transform(func=self._cost_encoder.transform_response_values)

    @property
    def cost_model(self) -> AbstractModel:
        """Returns the cost model."""
        return self._cost_model

    def predict_cost(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict evaluation costs in the original (non-transformed) scale.

        The model predicts in the encoder's transformed space (e.g. log-space).
        This method applies the inverse transform so that callers receive
        predictions in the original cost scale.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameters + #features]
            Input data points.

        Returns
        -------
        means : np.ndarray [#samples, 1]
            Predicted costs in original scale.
        vars : np.ndarray [#samples, 1] | None
            Predictive variance (left in transformed space; only meaningful
            for ranking, not for calibrated uncertainty in original scale).
        """
        mean, var = self._cost_model.predict(X)

        # Undo the encoder's response transform.
        # For RunHistoryLogEncoder this means exp(); for the identity
        # encoder (RunHistoryEncoder) this is a no-op.
        if isinstance(self._cost_encoder, RunHistoryLogEncoder):
            mean = np.exp(mean)

        mean = np.maximum(mean, 1e-9)
        return mean, var

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """Called after a trial completes.

        Extracts the cost, updates the dedicated cost RunHistory, and retrains the cost model.
        """
        evaluation_cost = value.additional_info.get("resource_cost", value.time)

        # Store the resource cost in the `cost` field so the encoder can process it.
        self._cost_runhistory.add(
            config=info.config,
            cost=evaluation_cost,
            time=value.time,
            status=value.status,
            seed=info.seed,
            budget=info.budget,
            instance=info.instance,
        )

        # Hand-crafted (formula-based) models don't learn from data.
        if isinstance(self._cost_model, HandCraftedCostModel):
            return None

        # The encoder's `transform()` returns raw (X, Y).
        # The model's y_transform (wired in __init__) applies the log
        # transform internally during `train()`.
        X, Y = self._cost_encoder.transform()

        if X.shape[0] > 0:
            self._cost_model.train(X, Y)

        return None
