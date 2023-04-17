from smac.random_design.modulus_design import (
    DynamicModulusRandomDesign,
    ModulusRandomDesign,
)

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def test_no_cool_down():
    c = ModulusRandomDesign(seed=1, modulus=3.0)
    for _ in range(10):
        assert c.check(1) is False
        assert c.check(2) is False
        assert c.check(3) is True
        assert c.check(4) is False
        assert c.check(5) is False
        assert c.check(6) is True
        assert c.check(30) is True
        c.next_iteration()

    c = ModulusRandomDesign(seed=1, modulus=1.0)
    for _ in range(10):
        assert c.check(1) is True
        assert c.check(2) is True
        assert c.check(30) is True
        c.next_iteration()


def test_linear_cool_down():
    c = DynamicModulusRandomDesign(seed=1, start_modulus=2.0, modulus_increment=1.0, end_modulus=4.0)
    for i in range(1, 100, 2):
        assert c.check(i) is False
        assert c.check(i + 1) is True

    c.next_iteration()
    for i in range(1, 100, 3):
        assert c.check(i) is False
        assert c.check(i + 1) is False
        assert c.check(i + 2) is True

    for i in range(10):
        c.next_iteration()
        for i in [1, 2, 3]:
            assert c.check(i) is False
        assert c.check(4) is True
        for i in [5, 6, 7]:
            assert c.check(i) is False
        assert c.check(8) is True
        # Repeat
        for i in [5, 6, 7]:
            assert c.check(i) is False
        assert c.check(8) is True
