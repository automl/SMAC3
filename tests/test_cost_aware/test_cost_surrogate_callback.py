"""Tests for the CostSurrogateCallback: the separate RunHistory mechanism,
log-transform pipeline (set_y_transform → train in log-space → predict_cost
in original scale), and HandCraftedCostModel bypass."""
from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac.callback.cost_surrogate_callback import CostSurrogateCallback
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.runhistory.dataclasses import TrialInfo, TrialValue, StatusType
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.runhistory.encoder.log_encoder import RunHistoryLogEncoder
from smac.scenario import Scenario


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)
    cs.add(UniformFloatHyperparameter("x", -5.0, 5.0, default_value=0))
    cs.add(UniformFloatHyperparameter("y", -5.0, 5.0, default_value=0))
    return cs


@pytest.fixture
def scenario(configspace, tmp_path) -> Scenario:
    return Scenario(
        configspace=configspace,
        name="CostSurrogateCallbackTest",
        objectives="cost",
        n_trials=100,
        seed=0,
        deterministic=True,
        output_directory=tmp_path,
    )


def _make_trial(configspace: ConfigurationSpace, seed: int = 0) -> tuple[TrialInfo, TrialValue, float]:
    """Create a trial with a known resource cost."""
    config = configspace.sample_configuration()
    x, y = config["x"], config["y"]
    resource_cost = abs(x) + abs(y) + 0.1  # always positive
    info = TrialInfo(config=config, seed=seed)
    value = TrialValue(
        cost=x**2 + y**2,  # performance — irrelevant for cost model
        time=0.1,
        status=StatusType.SUCCESS,
        additional_info={"resource_cost": resource_cost},
    )
    return info, value, resource_cost


# ---------------------------------------------------------------------------
# Tests: separate RunHistory stores resource cost in the `cost` field
# ---------------------------------------------------------------------------

def test_cost_runhistory_stores_resource_cost_not_performance(scenario, configspace):
    """The callback's internal RunHistory must contain resource costs (not
    performance) so that the encoder trains the cost model on the right values."""
    cost_model = HyperparameterOptimizationFacade.get_model(scenario)
    cb = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    infos, costs = [], []
    for _ in range(5):
        info, value, resource_cost = _make_trial(configspace)
        cb.on_tell_end(None, info, value)  # type: ignore[arg-type]
        infos.append(info)
        costs.append(resource_cost)

    # The cost RunHistory should have exactly the resource costs, not performance
    for trial_value, expected_cost in zip(cb._cost_runhistory.values(), costs):
        assert np.isclose(trial_value.cost, expected_cost), (
            f"Cost RunHistory has {trial_value.cost}, expected resource cost {expected_cost}"
        )


# ---------------------------------------------------------------------------
# Tests: y_transform is wired into the cost model
# ---------------------------------------------------------------------------

def test_set_y_transform_wired_for_learned_model(scenario):
    """For learned surrogates (e.g. RF), set_y_transform must be called so the
    model trains in log-space."""
    cost_model = HyperparameterOptimizationFacade.get_model(scenario)
    assert cost_model.transformer.y_transform is None, "Model should have no y_transform initially"

    cb = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    assert cost_model.transformer.y_transform is not None
    assert cost_model.transformer.y_transform == cb._cost_encoder.transform_response_values


def test_set_y_transform_not_wired_for_hand_crafted_model(scenario):
    """HandCraftedCostModel should not have y_transform set — it doesn't train."""
    cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=lambda cfg: 1.0)
    cb = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    assert cost_model.transformer.y_transform is None


# ---------------------------------------------------------------------------
# Tests: predict_cost returns original-scale values
# ---------------------------------------------------------------------------

def test_predict_cost_returns_original_scale(scenario, configspace):
    """predict_cost must return values in the original cost scale (not log-space),
    close to the training data."""
    cost_model = HyperparameterOptimizationFacade.get_model(scenario)
    cb = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    # Feed several trials with known costs
    known_costs = []
    for _ in range(10):
        info, value, resource_cost = _make_trial(configspace)
        cb.on_tell_end(None, info, value)  # type: ignore[arg-type]
        known_costs.append(resource_cost)

    # predict_cost should return positive values in the same order of magnitude
    # as the training costs (not log-scale values which would be much smaller)
    X_test = np.array([info.config.get_array()])
    pred_cost, _ = cb.predict_cost(X_test)

    assert pred_cost.shape == (1, 1)
    assert pred_cost[0, 0] > 0, "Predicted cost must be positive"

    # The prediction should be in the ballpark of actual costs,
    # not in log-space (which would be ~log(cost) ≈ much smaller or negative)
    min_cost, max_cost = min(known_costs), max(known_costs)
    assert pred_cost[0, 0] < max_cost * 10, (
        f"Predicted cost {pred_cost[0, 0]:.4f} is implausibly large vs max training cost {max_cost:.4f}"
    )


def test_predict_cost_matches_exp_of_raw_predict(scenario, configspace):
    """predict_cost should equal exp(model.predict) for the default log encoder."""
    cost_model = HyperparameterOptimizationFacade.get_model(scenario)
    cb = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    for _ in range(10):
        info, value, _ = _make_trial(configspace)
        cb.on_tell_end(None, info, value)  # type: ignore[arg-type]

    X_test = np.array([info.config.get_array()])

    # Raw predict returns log-space
    raw_mean, _ = cost_model.predict(X_test)
    # predict_cost returns exp(raw)
    cost_mean, _ = cb.predict_cost(X_test)

    expected = np.maximum(np.exp(raw_mean), 1e-9)
    np.testing.assert_allclose(cost_mean, expected, rtol=1e-6)


def test_predict_cost_identity_with_no_log_encoder(scenario, configspace):
    """When using a plain RunHistoryEncoder (no log), predict_cost should
    return the same values as model.predict (plus the 1e-9 floor)."""
    cost_model = HyperparameterOptimizationFacade.get_model(scenario)
    plain_encoder = RunHistoryEncoder(scenario=scenario)
    cb = CostSurrogateCallback(
        cost_model=cost_model, scenario=scenario, cost_encoder=plain_encoder,
    )

    for _ in range(10):
        info, value, _ = _make_trial(configspace)
        cb.on_tell_end(None, info, value)  # type: ignore[arg-type]

    X_test = np.array([info.config.get_array()])

    raw_mean, _ = cost_model.predict(X_test)
    cost_mean, _ = cb.predict_cost(X_test)

    # No log transform → predict_cost just applies the 1e-9 floor
    expected = np.maximum(raw_mean, 1e-9)
    np.testing.assert_allclose(cost_mean, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Tests: HandCraftedCostModel bypass
# ---------------------------------------------------------------------------

def test_hand_crafted_model_skips_retraining(scenario, configspace):
    """on_tell_end should not retrain a HandCraftedCostModel (it's formula-based)."""
    call_count = 0
    original_train = HandCraftedCostModel.train

    def mock_train(self, X, Y):
        nonlocal call_count
        call_count += 1
        return original_train(self, X, Y)

    cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=lambda cfg: 1.0)
    cost_model.train = lambda X, Y: mock_train(cost_model, X, Y)  # type: ignore

    cb = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    for _ in range(5):
        info, value, _ = _make_trial(configspace)
        cb.on_tell_end(None, info, value)  # type: ignore[arg-type]

    assert call_count == 0, "HandCraftedCostModel.train should never be called"


def test_hand_crafted_model_still_records_costs(scenario, configspace):
    """Even with a HandCraftedCostModel, the cost RunHistory should be populated."""
    cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=lambda cfg: 1.0)
    cb = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    for _ in range(3):
        info, value, _ = _make_trial(configspace)
        cb.on_tell_end(None, info, value)  # type: ignore[arg-type]

    assert len(cb._cost_runhistory) == 3


# ---------------------------------------------------------------------------
# Tests: default encoder is RunHistoryLogEncoder
# ---------------------------------------------------------------------------

def test_default_encoder_is_log_encoder(scenario):
    """The default cost encoder should be RunHistoryLogEncoder."""
    cost_model = HyperparameterOptimizationFacade.get_model(scenario)
    cb = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)

    assert isinstance(cb._cost_encoder, RunHistoryLogEncoder)


def test_custom_encoder_is_used(scenario):
    """A user-provided encoder should be used instead of the default."""
    cost_model = HyperparameterOptimizationFacade.get_model(scenario)
    custom_encoder = RunHistoryEncoder(scenario=scenario)
    cb = CostSurrogateCallback(
        cost_model=cost_model, scenario=scenario, cost_encoder=custom_encoder,
    )

    assert cb._cost_encoder is custom_encoder
