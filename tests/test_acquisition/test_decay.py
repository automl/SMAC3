import numpy as np
import pytest

from smac.acquisition.weight.decay import (
    DECAY_SHAPES,
    LogarithmicDecay,
    NoDecay,
    PolynomialDecay,
    get_decay_schedule,
)


def test_no_decay_is_exactly_one():
    """The exponent has to be exactly one, so that a weight without decay is left bit-identical."""
    schedule = NoDecay()

    for steps in (0, 1, 10, 1000):
        assert schedule(steps) == 1.0


def test_polynomial_decay_reproduces_pibo():
    """The piBO schedule is beta / (steps + 1)."""
    schedule = PolynomialDecay(beta=10.0)

    assert schedule(0) == pytest.approx(10.0)
    assert schedule(1) == pytest.approx(5.0)
    assert schedule(9) == pytest.approx(1.0)


@pytest.mark.parametrize("power", [1.0, 2.0, 3.0, 4.0, 5.0])
def test_polynomial_decay_follows_its_power(power):
    schedule = PolynomialDecay(beta=20.0, power=power)

    for steps in (0, 3, 17):
        assert schedule(steps) == pytest.approx(20.0 / (steps + 1) ** power)


def test_logarithmic_decay_is_the_slowest_shape():
    beta = 20.0
    logarithmic = LogarithmicDecay(beta)
    linear = PolynomialDecay(beta)

    assert logarithmic(0) == pytest.approx(beta / np.log(2))

    # Past the first few steps the logarithm shrinks far more slowly than the polynomial.
    for steps in (10, 50, 200):
        assert logarithmic(steps) > linear(steps)


def test_a_schedule_decays_monotonically():
    for schedule in (PolynomialDecay(20.0), PolynomialDecay(20.0, power=3.0), LogarithmicDecay(20.0)):
        exponents = [schedule(steps) for steps in range(50)]

        assert exponents == sorted(exponents, reverse=True)


def test_negative_steps_are_clamped():
    """A weight anchored in the future must not blow the exponent up."""
    schedule = PolynomialDecay(beta=10.0)

    assert schedule(-5) == schedule(0)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="decay factor must be positive"):
        PolynomialDecay(beta=0.0)

    with pytest.raises(ValueError, match="decay power must be positive"):
        PolynomialDecay(beta=1.0, power=-1.0)

    with pytest.raises(ValueError, match="decay offset must be positive"):
        PolynomialDecay(beta=1.0, offset=0.0)

    with pytest.raises(ValueError, match="decay factor must be positive"):
        LogarithmicDecay(beta=-1.0)


def test_meta_describes_the_schedule():
    assert NoDecay().meta == {"name": "NoDecay"}
    assert PolynomialDecay(20.0, power=2.0).meta == {
        "name": "PolynomialDecay",
        "beta": 20.0,
        "power": 2.0,
        "offset": 1.0,
    }
    assert LogarithmicDecay(20.0).meta == {"name": "LogarithmicDecay", "beta": 20.0, "offset": 1.0}


def test_shapes_can_be_selected_by_name():
    assert sorted(DECAY_SHAPES) == ["cubic", "linear", "logarithmic", "quadratic", "quartic", "quintic"]

    for shape in DECAY_SHAPES:
        schedule = get_decay_schedule(shape, beta=20.0)
        assert schedule(0) > 0

    assert get_decay_schedule("quadratic", 20.0)(3) == pytest.approx(20.0 / 16.0)

    with pytest.raises(ValueError, match="Unknown decay shape"):
        get_decay_schedule("exponential", 20.0)
