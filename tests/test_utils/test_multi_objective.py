import pytest

from smac.utils.multi_objective import normalize_costs


def test_returns_input_when_bounds_none():
    values = [0.2, 0.5, 0.9]
    assert normalize_costs(values, None) == values


def test_mismatched_lengths_raises_value_error():
    with pytest.raises(ValueError):
        normalize_costs([1.0, 2.0], [(0.0, 2.0)])


def test_normalization_basic():
    values = [1.0, 2.0, 3.0]
    bounds = [(0.0, 10.0), (0.0, 4.0), (1.0, 5.0)]
    out = normalize_costs(values, bounds)
    # Expected: (1-0)/10 = 0.1, (2-0)/4 = 0.5, (3-1)/4 = 0.5
    assert out == pytest.approx([0.1, 0.5, 0.5])


@pytest.mark.parametrize(
    "value,bounds,expected",
    [
        (-10.0, (0.0, 5.0), 0.0),
        (10.0, (0.0, 5.0), 1.0),
        (0.0, (-5.0, 5.0), 0.5),
    ],
)
def test_clamping_and_negative_ranges(value, bounds, expected):
    out = normalize_costs([value], [bounds])
    assert out == pytest.approx([expected])


def test_identical_bounds_sets_to_one():
    # When min == max, result must be 1.0 regardless of value
    values = [0.0, 1.0, 100.0]
    bounds = [(5.0, 5.0)] * 3
    out = normalize_costs(values, bounds)
    assert out == [1.0, 1.0, 1.0]


def test_almost_identical_bounds_threshold():
    b0 = 1.0
    b1 = 1.0 + 1e-12
    out = normalize_costs([1.0, 1.0 + 5e-13, 2.0], [(b0, b1)] * 3)
    assert out == [1.0, 1.0, 1.0]


def test_assertion_when_value_is_list():
    # Function asserts each v is not a list
    with pytest.raises(AssertionError):
        normalize_costs([[1.0]], [(0.0, 2.0)])