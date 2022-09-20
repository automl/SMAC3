import pytest

from smac.utils.multi_objective import normalize_costs

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


@pytest.fixture
def bounds():
    return [(0, 50), (50, 100)]


@pytest.fixture
def bounds_invalid():
    return [(0, 0), (5, 5)]


def test_normalize_costs(bounds, bounds_invalid):
    # If no bounds are passed, we get the same result back
    v = [5, 2]
    nv = normalize_costs(v)
    assert nv == [5, 2]

    # Normalize between 0..1 given data only
    v = [25, 50]
    nv = normalize_costs(v, bounds)
    assert nv == [0.5, 0]

    # Invalid bounds
    v = [25, 50]
    nv = normalize_costs(v, bounds_invalid)
    assert nv == [1, 1]

    # Invalid input
    v = [[25], [50]]
    with pytest.raises(AssertionError):
        nv = normalize_costs(v, bounds)

    # Wrong shape
    v = [25, 50, 75]
    with pytest.raises(ValueError):
        nv = normalize_costs(v, bounds)
