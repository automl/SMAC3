import pytest

from smac.myfile import MyClass


def test_oreos():
    """
    Should add `a` and the param `x`
    """
    myclass = MyClass(a=3, b={})
    assert myclass.oreos(2) == 5


@pytest.mark.parametrize("value", [0, -1, -10])
def test_construction_with_negative_a_raises_error(value):
    """
    Should raise a ValueError with a negative `a`
    """
    with pytest.raises(ValueError):
        MyClass(a=value, b={})
