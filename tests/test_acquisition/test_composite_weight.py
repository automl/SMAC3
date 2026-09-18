import numpy as np
import pytest
from test_weighted_acquisition_function import ConstantWeight, Model

from smac.acquisition.weight import CompositeWeight, PolynomialDecay


@pytest.fixture
def X():
    return np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])


def _updated(weight, **kwargs):
    kwargs.setdefault("eta", 1.0)
    kwargs.setdefault("num_data", 10)
    weight.update(model=Model(), **kwargs)

    return weight


@pytest.mark.parametrize(
    "combination, expected",
    [("sum", 0.5 + 0.2), ("product", 0.5 * 0.2), ("max", 0.5)],
)
def test_members_are_combined_by_the_chosen_rule(X, combination, expected):
    composite = _updated(
        CompositeWeight([ConstantWeight(0.5, floor=0.0), ConstantWeight(0.2, floor=0.0)], combination=combination)
    )

    assert composite(X) == pytest.approx(np.full((len(X), 1), expected))


def test_summing_is_the_default(X):
    assert CompositeWeight().combination == "sum"


def test_an_unknown_combination_is_rejected():
    with pytest.raises(ValueError, match="Unknown combination"):
        CompositeWeight(combination="average")


def test_a_composite_of_one_member_reproduces_that_member(X):
    """A sum of one term is a product of one term, which is what keeps piBO exact under the ensemble."""
    member = ConstantWeight(0.3, floor=1e-3, decay=PolynomialDecay(beta=4.0))
    composite = _updated(CompositeWeight([member]))

    assert composite(X) == pytest.approx(member(X))


def test_each_member_decays_from_its_own_anchor(X):
    """The point of the composite: members which arrived at different times carry different influence."""
    early = ConstantWeight(0.5, floor=0.0, decay=PolynomialDecay(beta=4.0))
    late = ConstantWeight(0.5, floor=0.0, decay=PolynomialDecay(beta=4.0))

    early._steps = 30
    late._steps = 0

    composite = CompositeWeight([early, late], combination="sum")

    # The older member has decayed towards one; the newer one is still close to its raw value.
    assert early(X)[0, 0] > late(X)[0, 0]
    assert composite(X) == pytest.approx(early(X) + late(X))


def test_members_are_keyed_and_can_be_managed():
    composite = CompositeWeight()

    first = ConstantWeight(0.5)
    key = composite.add(first, key="prior")

    assert key == "prior"
    assert composite.members == {"prior": first}
    assert len(composite) == 1

    second = ConstantWeight(0.2)
    composite.replace("prior", second)
    assert composite.members == {"prior": second}

    assert composite.remove("prior") is second
    assert len(composite) == 0


def test_keys_are_generated_when_not_given():
    composite = CompositeWeight()

    first = composite.add(ConstantWeight())
    second = composite.add(ConstantWeight())

    assert first != second


def test_a_duplicate_key_is_rejected():
    composite = CompositeWeight()
    composite.add(ConstantWeight(), key="prior")

    with pytest.raises(ValueError, match="already registered"):
        composite.add(ConstantWeight(), key="prior")


def test_removing_an_unknown_key_is_rejected():
    with pytest.raises(KeyError, match="No weight is registered"):
        CompositeWeight().remove("prior")

    with pytest.raises(KeyError, match="No weight is registered"):
        CompositeWeight().replace("prior", ConstantWeight())


def test_a_member_added_later_is_brought_up_to_date_immediately():
    """A prior injected between two iterations has to be usable straight away."""
    composite = _updated(CompositeWeight(), num_data=17)

    added = ConstantWeight()
    composite.add(added)

    assert len(added.updates) == 1
    assert added.updates[0]["num_data"] == 17
    assert added.model is not None


def test_an_empty_composite_stands_aside(X):
    composite = _updated(CompositeWeight())

    assert composite.is_active() is False
    assert composite.suppresses_acquisition() is False
    assert composite.adjust_eta(3.0) == 3.0


def test_inactive_members_are_left_out_of_the_combination(X):
    composite = _updated(
        CompositeWeight(
            [ConstantWeight(0.5, floor=0.0), ConstantWeight(0.2, floor=0.0, active=False)], combination="sum"
        )
    )

    assert composite.is_active() is True
    assert composite(X) == pytest.approx(np.full((len(X), 1), 0.5))


def test_the_composite_relays_the_hooks_of_its_members():
    composite = CompositeWeight([ConstantWeight(suppresses=True), ConstantWeight(eta=9.0)])

    assert composite.suppresses_acquisition() is True
    assert composite.adjust_eta(1.0) == 9.0


def test_clearing_removes_every_member():
    composite = CompositeWeight([ConstantWeight(), ConstantWeight()])
    composite.clear()

    assert len(composite) == 0


def test_updating_reaches_every_member():
    members = [ConstantWeight(), ConstantWeight()]
    _updated(CompositeWeight(members), num_data=5)

    for member in members:
        assert len(member.updates) == 1
        assert member.updates[0]["num_data"] == 5


def test_meta_describes_the_members():
    composite = CompositeWeight({"prior": ConstantWeight()}, combination="product")
    meta = composite.meta

    assert meta["name"] == "CompositeWeight"
    assert meta["combination"] == "product"
    assert meta["weights"]["prior"]["name"] == "ConstantWeight"
