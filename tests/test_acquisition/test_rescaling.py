import numpy as np
import pytest

from smac.acquisition.function import (
    EI,
    LCB,
    PI,
    TS,
    ConstrainedAcquisitionFunction,
    IntegratedAcquisitionFunction,
    PriorAcquisitionFunction,
)
from smac.acquisition.function.confidence_bound import UCB
from smac.utils.constraints import OutcomeConstraint


class _Model:
    """Just enough of a model to construct the wrappers."""

    @property
    def meta(self):
        return {"name": "_Model"}

    def train(self, X, Y):
        return self

    def predict_marginalized(self, X):
        return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))


@pytest.mark.parametrize(
    "acquisition_function, expected",
    [(EI(), False), (PI(), False), (LCB(), True), (UCB(), True), (TS(), True)],
)
def test_only_negative_acquisition_functions_require_rescaling(acquisition_function, expected):
    assert acquisition_function.requires_rescaling is expected


@pytest.mark.parametrize("acquisition_function", [EI(), LCB(), UCB(), TS()])
def test_integration_delegates_the_decision(acquisition_function):
    integrated = IntegratedAcquisitionFunction(acquisition_function)

    assert integrated.requires_rescaling is acquisition_function.requires_rescaling


@pytest.mark.parametrize("wrapped, expected", [(LCB(), True), (UCB(), True), (TS(), True), (EI(), False)])
def test_the_prior_wrapper_rescales_every_negative_acquisition_function(wrapped, expected):
    """UCB is as negative as LCB, so it has to be rescaled just the same.

    Multiplying a negative acquisition value by a prior density in [0, 1] moves it up, so without the shift the
    prior would favour exactly the configurations it is supposed to discourage.
    """
    assert PriorAcquisitionFunction(wrapped, decay_beta=1.0)._rescale is expected


@pytest.mark.parametrize("wrapped, expected", [(LCB(), True), (UCB(), True), (TS(), True), (EI(), False)])
def test_the_constrained_wrapper_rescales_every_negative_acquisition_function(wrapped, expected):
    constrained = ConstrainedAcquisitionFunction(
        acquisition_function=wrapped,
        constraints=[OutcomeConstraint("latency", "<=", 100.0)],
        constraint_model=_Model(),
    )

    assert constrained._rescale is expected


@pytest.mark.parametrize("wrapped", [LCB(), UCB(), TS(), EI()])
def test_the_wrappers_agree_on_rescaling(wrapped):
    """The two wrappers used to disagree: the prior one missed UCB."""
    constrained = ConstrainedAcquisitionFunction(
        acquisition_function=wrapped,
        constraints=[OutcomeConstraint("latency", "<=", 100.0)],
        constraint_model=_Model(),
    )

    assert PriorAcquisitionFunction(wrapped, decay_beta=1.0)._rescale is constrained._rescale
