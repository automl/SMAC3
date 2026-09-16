from __future__ import annotations

from smac.callback import Callback
from smac.main.smbo import SMBO
from smac.model.abstract_model import AbstractModel
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runhistory.encoder.abstract_encoder import AbstractRunHistoryEncoder
from smac.runhistory.encoder.log_scaled_encoder import RunHistoryLogScaledEncoder
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class CostSurrogateCallback(Callback):
    """
    Callback to train a cost model on completed trials. This is essential for the
    main optimization loop after the initial design is finished.

    Parameters
    ----------
    cost_model : AbstractModel
        The surrogate model to be trained on the cost data.
    scenario : Scenario
        The SMAC scenario object.
    encoder : AbstractRunHistoryEncoder | None, defaults to None
        The encoder to transform the cost data before training the model.
        If None, defaults to `RunHistoryLogScaledEncoder`.
    """

    _encoder: AbstractRunHistoryEncoder

    def __init__(
        self,
        cost_model: AbstractModel,
        scenario: Scenario,
        encoder: AbstractRunHistoryEncoder | None = None,
    ):
        self._scenario = scenario
        self._cost_model = cost_model
        self._cost_runhistory = RunHistory()

        if encoder is None:
            self._encoder = RunHistoryLogScaledEncoder(scenario=scenario)
        else:
            self._encoder = encoder

        self._encoder.runhistory = self._cost_runhistory

    @property
    def cost_model(self) -> AbstractModel:
        """Returns the cost model."""
        return self._cost_model

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """
        This method is called after a real trial is completed. It extracts the cost,
        updates the history, and retrains the cost model with real data.
        """
        evaluation_cost = value.additional_info.get("resource_cost", value.time)

        # Add the trial to our dedicated cost runhistory.
        # We put the `evaluation_cost` into the `cost` field so the encoder can process it.
        self._cost_runhistory.add(
            config=info.config,
            cost=evaluation_cost,
            time=value.time,
            status=value.status,
            seed=info.seed,
            budget=info.budget,
            instance=info.instance,
        )

        # Retrain the model on all available data in our cost runhistory
        # The transform method will now correctly calculate statistics on the evaluation costs.
        X, Y_scaled = self._encoder.transform()

        if X.shape[0] > 0:
            self._cost_model.train(X, Y_scaled)

        return None
