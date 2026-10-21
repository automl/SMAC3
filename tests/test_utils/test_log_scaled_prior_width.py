import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float, Integer

from smac.utils.configspace import create_prior_configspace_copy


def _geometric_spread(lower: float, upper: float, mu: float, std_denominator: float) -> float:
    """Samples a log-scaled prior built from ``mu`` and returns its measured geometric standard deviation.

    2000 draws pin the spread down to about 2%, well inside the tolerances the tests use.
    """
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Float("x", (lower, upper), log=True))
    prior = create_prior_configspace_copy(configspace, {"x": mu}, std_denominator=std_denominator)
    drawn = np.array([configuration["x"] for configuration in prior.sample_configuration(2000)])

    return float(np.exp(np.std(np.log(drawn))))


@pytest.mark.parametrize(
    "lower, upper, std_denominator",
    [(1e-5, 1e-1, 4.0), (1e-4, 1.0, 4.0), (1.0, 5.0, 4.0), (1.0, 21.0, 20.0), (16.0, 512.0, 4.0)],
)
def test_log_scaled_sigma_is_a_factor_spanning_the_logarithmic_range(lower, upper, std_denominator):
    """A log-scaled belief gets sigma = (upper / lower) ** (1 / std_denominator), which is always above one.

    On a log scale ConfigSpace reads sigma as a multiplicative factor, so this is the factor whose logarithm is
    the logarithmic range divided by ``std_denominator``. The cases include [1, 5] and [1, 21], where
    ``upper - lower`` equals ``std_denominator``: a linear-scale sigma there is exactly one, a factor of one,
    which is no spread at all.
    """
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Float("x", (lower, upper), log=True))

    prior = create_prior_configspace_copy(configspace, {"x": lower}, std_denominator=std_denominator)
    sigma = dict(prior)["x"].sigma

    assert sigma == pytest.approx((upper / lower) ** (1 / std_denominator))
    assert sigma > 1


def test_log_scaled_integer_sigma_is_a_factor():
    """A log-scaled integer belief uses the same factor as a log-scaled float one.

    Setup: an integer on [16, 512] with the default denominator of 4.
    Expect: sigma is 32 ** (1 / 4), not the linear 124.
    """
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Integer("n", (16, 512), log=True))

    prior = create_prior_configspace_copy(configspace, {"n": 64})

    assert dict(prior)["n"].sigma == pytest.approx(32 ** (1 / 4))


def test_log_scaled_belief_has_the_requested_spread():
    """Samples from a log-scaled belief spread by the factor the denominator asks for.

    Setup: [1e-6, 1e-2] with a denominator of 20, believed at the geometric centre 1e-4. The requested factor is
    (1e4) ** (1 / 20), about 1.585; centring the belief and keeping it narrow leaves the truncation at the bounds
    ten standard deviations away, so the measured spread is the untruncated one.
    Expect: the measured geometric standard deviation is within 5% of 1.585.
    """
    spread = _geometric_spread(1e-6, 1e-2, 1e-4, std_denominator=20.0)

    assert spread == pytest.approx(1e4 ** (1 / 20), rel=0.05)


def test_a_larger_denominator_sharpens_a_log_scaled_belief():
    """Raising the denominator narrows a log-scaled belief, as it does a linear one.

    Setup: [1e-4, 1] believed at its geometric centre 1e-2, with denominators 4 and 20.
    Expect: the measured spread with 20 is smaller than with 4.
    """
    wide = _geometric_spread(1e-4, 1.0, 1e-2, std_denominator=4.0)
    narrow = _geometric_spread(1e-4, 1.0, 1e-2, std_denominator=20.0)

    assert narrow < wide
