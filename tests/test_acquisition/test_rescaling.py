"""Every acquisition function declares which convention its values follow.

`AcquisitionScale` is what anything weighting an acquisition function consults before it touches a value,
and the three answers call for three different arithmetic - multiply, add, or shift and then multiply.
Getting it wrong is silent: the search keeps running and steers the wrong way, which is what these pin.

The file is named for the bug it was written against. The two wrappers each decided for themselves whether a
value needed shifting, and they disagreed - the prior one missed UCB, which is exactly as negative as LCB, so
a prior applied to UCB favoured the configurations it was meant to discourage. There is one decision now, and
the last test here is what keeps it that way.
"""

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
from smac.acquisition.function.abstract_acquisition_function import AcquisitionScale
from smac.acquisition.function.confidence_bound import UCB
from smac.acquisition.function.log_expected_improvement import LogEI
from smac.utils.constraints import OutcomeConstraint

LINEAR = AcquisitionScale.LINEAR
LOG = AcquisitionScale.LOG
SIGNED = AcquisitionScale.SIGNED


class _Model:
    """Just enough of a model to construct the wrappers."""

    @property
    def meta(self):
        return {"name": "_Model"}

    def train(self, X, Y):
        return self

    def predict_marginalized(self, X):
        return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))


def _constrained(wrapped):
    return ConstrainedAcquisitionFunction(
        acquisition_function=wrapped,
        constraints=[OutcomeConstraint("latency", "<=", 100.0)],
        constraint_model=_Model(),
    )


@pytest.mark.parametrize(
    "acquisition_function, expected",
    [(EI(), LINEAR), (PI(), LINEAR), (LogEI(), LOG), (LCB(), SIGNED), (UCB(), SIGNED), (TS(), SIGNED)],
)
def test_every_acquisition_function_declares_its_scale(acquisition_function, expected):
    assert acquisition_function.value_scale is expected


def test_the_scales_are_mutually_exclusive():
    """The reason this is one property and not two booleans.

    `LogEI` is negative, and so is `LCB`, but for different reasons calling for opposite treatment: one is a
    logarithm and wants its weights added, the other is a cost and wants shifting first. Two flags would let
    something claim both, which describes nothing.
    """
    assert len({LINEAR, LOG, SIGNED}) == 3
    assert LogEI().value_scale is not LCB().value_scale


@pytest.mark.parametrize("acquisition_function", [EI(), LogEI(), LCB(), UCB(), TS()])
def test_integration_delegates_the_decision(acquisition_function):
    integrated = IntegratedAcquisitionFunction(acquisition_function)

    assert integrated.value_scale is acquisition_function.value_scale


@pytest.mark.parametrize(
    "wrapped, expected",
    [(LCB(), SIGNED), (UCB(), SIGNED), (TS(), SIGNED), (EI(), LINEAR), (LogEI(), LOG)],
)
def test_the_prior_wrapper_reads_the_scale_of_what_it_wraps(wrapped, expected):
    assert PriorAcquisitionFunction(wrapped, decay_beta=1.0)._scale is expected


@pytest.mark.parametrize(
    "wrapped, expected",
    [(LCB(), SIGNED), (UCB(), SIGNED), (TS(), SIGNED), (EI(), LINEAR), (LogEI(), LOG)],
)
def test_the_constrained_wrapper_reads_the_scale_of_what_it_wraps(wrapped, expected):
    assert _constrained(wrapped)._scale is expected


@pytest.mark.parametrize("wrapped", [LCB(), UCB(), TS(), EI(), LogEI()])
def test_the_wrappers_agree(wrapped):
    """Both are shims over the same weighting layer, so they cannot disagree any more - but this is the test
    that failed when they were two implementations, so it stays."""
    assert PriorAcquisitionFunction(wrapped, decay_beta=1.0)._scale is _constrained(wrapped)._scale


@pytest.mark.parametrize("wrapped, expected", [(LCB(), LINEAR), (EI(), LINEAR), (LogEI(), LOG)])
def test_what_comes_out_of_a_wrapper(wrapped, expected):
    """A shifted signed function leaves on the linear convention; a logarithmic one stays logarithmic.

    This is what lets a wrapper be wrapped by something that weights again without the shift being applied
    twice - the bug that nesting used to cause.
    """
    assert _constrained(wrapped).value_scale is expected
