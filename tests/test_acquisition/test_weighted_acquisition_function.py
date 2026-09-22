import numpy as np
import pytest

from smac.acquisition.function import EI, LCB, WeightedAcquisitionFunction
from smac.acquisition.weight import CompositeWeight, NoDecay, PolynomialDecay
from smac.acquisition.weight.abstract_weight import AbstractAcquisitionWeight
from smac.acquisition.function.abstract_acquisition_function import AcquisitionScale


class ConstantWeight(AbstractAcquisitionWeight):
    """A weight which returns the same value everywhere, with every hook made settable."""

    def __init__(self, value=0.5, *, active=True, suppresses=False, eta=None, **kwargs):
        super().__init__(**kwargs)
        self._value = value
        self._active = active
        self._suppresses = suppresses
        self._eta = eta
        self.updates = []

    def _compute(self, X):
        return np.full((X.shape[0], 1), self._value)

    def _update(self, **kwargs):
        self.updates.append(kwargs)

    def is_active(self):
        return self._active

    def suppresses_acquisition(self):
        return self._suppresses

    def adjust_eta(self, eta):
        return self._eta if self._eta is not None else eta


class LinearAcquisition(EI):
    """An acquisition function whose values are the first column of X, so they are trivial to predict."""

    @property
    def value_scale(self):
        return AcquisitionScale.LINEAR

    def _update(self, **kwargs):
        pass

    def _compute(self, X):
        return X[:, 0].reshape((-1, 1))


class NegativeAcquisition(LinearAcquisition):
    """Negative by construction, like a confidence bound."""

    @property
    def value_scale(self):
        return AcquisitionScale.SIGNED


class Model:
    @property
    def meta(self):
        return {"name": "Model"}


@pytest.fixture
def X():
    return np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [4.0, 1.0]])


def _updated(acquisition_function, **kwargs):
    kwargs.setdefault("eta", 1.0)
    kwargs.setdefault("num_data", 10)
    acquisition_function.update(model=Model(), **kwargs)

    return acquisition_function


def test_weights_multiply_into_the_acquisition_value(X):
    weighted = _updated(WeightedAcquisitionFunction(LinearAcquisition(), [ConstantWeight(0.5), ConstantWeight(0.2)]))

    # floor 1e-12 on each weight, so the product is not exactly 0.1
    expected = X[:, 0].reshape((-1, 1)) * (0.5 + 1e-12) * (0.2 + 1e-12)

    assert weighted._compute(X) == pytest.approx(expected)


def test_the_result_is_always_a_column(X):
    weighted = _updated(WeightedAcquisitionFunction(LinearAcquisition(), [ConstantWeight()]))

    assert weighted._compute(X).shape == (len(X), 1)


def test_without_weights_the_acquisition_function_passes_through_untouched(X):
    inner = LinearAcquisition()
    weighted = _updated(WeightedAcquisitionFunction(inner, []))

    assert np.array_equal(weighted._compute(X), inner._compute(X))


def test_an_inactive_weight_does_not_even_rescale(X):
    """An inactive weight is skipped entirely, not treated as a constant one.

    A constant weight would still force the rescaling, which clips at zero and changes the ranking.
    """
    inner = NegativeAcquisition()
    weighted = _updated(WeightedAcquisitionFunction(inner, [ConstantWeight(active=False)]), eta=-100.0)

    assert np.array_equal(weighted._compute(X), inner._compute(X))


def test_a_negative_acquisition_function_is_shifted_before_weighting(X):
    weighted = _updated(WeightedAcquisitionFunction(NegativeAcquisition(), [ConstantWeight(0.5)]), eta=3.0)

    expected = np.clip(X[:, 0].reshape((-1, 1)) + 3.0, 0, np.inf) * (0.5 + 1e-12)

    assert weighted._compute(X) == pytest.approx(expected)


def test_the_shift_is_applied_exactly_once(X):
    """Two weights must not shift the values twice; that is the bug nesting wrappers used to cause."""
    one = _updated(WeightedAcquisitionFunction(NegativeAcquisition(), [ConstantWeight(1.0)]), eta=3.0)
    two = _updated(
        WeightedAcquisitionFunction(NegativeAcquisition(), [ConstantWeight(1.0), ConstantWeight(1.0)]), eta=3.0
    )

    assert two._compute(X) == pytest.approx(one._compute(X))


def test_a_wrapper_refuses_to_wrap_another_wrapper():
    inner = WeightedAcquisitionFunction(LinearAcquisition())

    with pytest.raises(ValueError, match="must not wrap another one"):
        WeightedAcquisitionFunction(inner)


def test_a_suppressing_weight_replaces_the_acquisition_value(X):
    weighted = _updated(
        WeightedAcquisitionFunction(LinearAcquisition(), [ConstantWeight(0.25, suppresses=True)]),
    )

    assert weighted._compute(X) == pytest.approx(np.full((len(X), 1), 0.25 + 1e-12))


def test_a_suppressing_weight_still_lets_the_others_speak(X):
    """The acquisition term is replaced by one, so the other weights keep guiding the search."""
    weighted = _updated(
        WeightedAcquisitionFunction(LinearAcquisition(), [ConstantWeight(0.25, suppresses=True), ConstantWeight(0.5)]),
    )

    assert weighted._compute(X) == pytest.approx(np.full((len(X), 1), (0.25 + 1e-12) * (0.5 + 1e-12)))


def test_weights_may_correct_the_incumbent_in_order():
    inner = LinearAcquisition()
    weighted = _updated(WeightedAcquisitionFunction(inner, [ConstantWeight(eta=5.0), ConstantWeight(eta=7.0)]), eta=1.0)

    assert weighted._eta == 7.0


def test_the_weights_are_updated_before_the_acquisition_function():
    weight = ConstantWeight()
    _updated(WeightedAcquisitionFunction(LinearAcquisition(), [weight]), eta=2.0, num_data=42)

    assert len(weight.updates) == 1
    assert weight.updates[0]["num_data"] == 42
    assert weight.updates[0]["eta"] == 2.0


def test_the_model_reaches_every_weight():
    weight = ConstantWeight()
    weighted = WeightedAcquisitionFunction(LinearAcquisition(), [weight])
    model = Model()
    weighted.model = model

    assert weight.model is model

    added = ConstantWeight()
    weighted.add_weight(added)

    assert added.model is model


def test_weights_can_be_added_and_removed():
    weight = ConstantWeight()
    weighted = WeightedAcquisitionFunction(LinearAcquisition())

    weighted.add_weight(weight)
    assert weighted.weights == (weight,)

    weighted.remove_weight(weight)
    assert weighted.weights == ()


def test_a_weight_can_be_found_by_its_type():
    composite = CompositeWeight()
    weighted = WeightedAcquisitionFunction(LinearAcquisition(), [ConstantWeight(), composite])

    assert weighted.get_weight(CompositeWeight) is composite
    assert weighted.get_weight(LinearAcquisition) is None


def test_a_weighted_acquisition_function_never_needs_rescaling():
    assert WeightedAcquisitionFunction(LCB()).value_scale is AcquisitionScale.LINEAR
    assert WeightedAcquisitionFunction(EI()).value_scale is AcquisitionScale.LINEAR


def test_the_floor_is_applied_before_the_decay(X):
    """Zero raised to any positive power is still zero, so flooring afterwards would lose the ranking."""
    weight = ConstantWeight(0.0, floor=1e-3, decay=PolynomialDecay(beta=2.0))
    weight.update(model=Model(), eta=1.0, num_data=1)

    assert np.all(weight(X) > 0)
    assert weight(X) == pytest.approx(np.full((len(X), 1), (0.0 + 1e-3) ** 2.0))


def test_no_decay_leaves_the_weight_exactly_alone(X):
    weight = ConstantWeight(0.3, floor=0.0, decay=NoDecay())
    weight.update(model=Model(), eta=1.0, num_data=1)

    assert np.all(weight(X) == 0.3)


def test_a_negative_floor_is_rejected():
    with pytest.raises(ValueError, match="floor must not be negative"):
        ConstantWeight(floor=-1.0)


def test_meta_describes_the_wrapping():
    weighted = WeightedAcquisitionFunction(EI(), [ConstantWeight()])
    meta = weighted.meta

    assert meta["name"] == "WeightedAcquisitionFunction"
    assert meta["acquisition_function"]["name"] == "EI"
    assert meta["weights"][0]["name"] == "ConstantWeight"
    assert meta["weights"][0]["decay"] == {"name": "NoDecay"}
