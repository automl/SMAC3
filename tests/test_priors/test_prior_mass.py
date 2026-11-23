import numpy as np
import pytest
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer, Normal

from smac.acquisition.weight import (
    ConfigSpacePrior,
    PolynomialDecay,
    PriorEnsemble,
    PriorWeight,
    TabulatedPrior,
)

GRID = np.linspace(0.0, 1.0, 200001).reshape(-1, 1)


class Model:
    """A stand-in surrogate; prior weights only need one to be handed over."""

    @property
    def meta(self):
        return {"name": "Model"}


def _space(sd: float) -> ConfigurationSpace:
    """One float on [0, 1] with a normal belief of standard deviation ``sd`` at its centre."""
    return ConfigurationSpace({"x": Float("x", (0.0, 1.0), distribution=Normal(0.5, sd))})


def _weight(space: ConfigurationSpace, steps: int, *, beta: float = 1.0, floor: float = 0.0) -> PriorWeight:
    """A prior weight anchored at trial 0 and updated to ``steps`` trials, so its exponent is beta / (steps + 1)."""
    weight = PriorWeight(ConfigSpacePrior(space), PolynomialDecay(beta=beta), t0=0, floor=floor)
    weight.update(model=Model(), eta=1.0, num_data=steps)

    return weight


def _mass(values: np.ndarray) -> float:
    """Integrates values on `GRID` over [0, 1]."""
    return float(np.trapezoid(values.ravel(), GRID.ravel()))


def test_a_configspace_prior_has_unit_mass_at_full_strength():
    """A ConfigSpace prior's mass at exponent one is one, sharp or broad, since ConfigSpace densities are normalized.

    The sharp case (sd 0.001) is a spike a thousandth of the range wide, which is what the integration has to find.
    """
    for sd in (0.001, 0.2):
        assert ConfigSpacePrior(_space(sd)).log_mass(1.0) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("steps", [0, 1, 4, 9, 99])
def test_a_normalized_belief_has_unit_mass_at_every_stage_of_its_decay(steps):
    """The normalized form integrates to one at every decay exponent, for a sharp and a broad belief alike.

    Setup: beta = 1, so after `steps` trials the exponent is 1 / (steps + 1), from 1 down to 0.01; no floor.
    Expect: both beliefs integrate to one within 0.1%. Unnormalized, the sharp one drops to about 7% at exponent 0.5.
    """
    for sd in (0.001, 0.2):
        weight = _weight(_space(sd), steps)

        assert _mass(weight.normalized(GRID)) == pytest.approx(1.0, rel=1e-3)


def test_a_faded_belief_becomes_flat():
    """At a vanishing exponent the normalized belief approaches the uniform density, one on the unit interval.

    Setup: a belief with sd 0.05 after 9999 trials, exponent 1e-4. At the bounds, ten standard deviations out, its
    density is about 1e-21: small, but not underflowed to zero, which would stay zero under any power.
    Expect: within 1% of one at the bounds, at the peak and in between.
    """
    weight = _weight(_space(0.05), 9999)
    X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])

    assert weight.normalized(X) == pytest.approx(np.ones((5, 1)), rel=1e-2)


def test_a_sharp_and_a_broad_belief_contribute_equal_mass_to_an_ensemble():
    """Summed in an ensemble, a sharp and a broad belief contribute the same mass, even half way through decay.

    Setup: two beliefs anchored at trial 0 and updated to trial 1, exponent 0.5, no floor. The ensemble integrates
    to two, one per belief.
    Expect: each member's normalized contribution integrates to one, and the ensemble to two.
    """
    sharp = _weight(_space(0.001), 1)
    broad = _weight(_space(0.2), 1)
    ensemble = PriorEnsemble({"sharp": sharp, "broad": broad})

    assert _mass(sharp.normalized(GRID)) == pytest.approx(_mass(broad.normalized(GRID)), rel=1e-3)
    assert _mass(ensemble(GRID)) == pytest.approx(2.0, rel=1e-3)


def test_a_single_belief_keeps_the_pibo_form():
    """With one member the ensemble applies the plain piBO weight, since dividing by a mass only rescales it.

    Expect: the ensemble equals the member's own (pi + floor) ** exponent exactly.
    """
    weight = _weight(_space(0.05), 3, beta=2.0, floor=1e-3)
    ensemble = PriorEnsemble([weight])
    X = np.array([[0.1], [0.5], [0.9]])

    assert ensemble(X) == pytest.approx(weight(X))


def test_scaling_a_table_does_not_change_the_belief():
    """Multiplying every density in a table by a constant leaves the normalized belief unchanged.

    Setup: the same two-knot table at scale one and at scale one thousand, decayed to exponent 0.5.
    Expect: identical normalized values. The raw `pdf` differs by the factor, as before.
    """
    space = ConfigurationSpace({"x": Float("x", (0.0, 1.0))})
    knots = [[0.2, 1.0], [0.8, 3.0]]
    small = PriorWeight(TabulatedPrior(space, {"x": knots}), PolynomialDecay(beta=1.0), t0=0, floor=0.0)
    large = PriorWeight(
        TabulatedPrior(space, {"x": [[p, 1000.0 * d] for p, d in knots]}), PolynomialDecay(beta=1.0), t0=0, floor=0.0
    )
    for weight in (small, large):
        weight.update(model=Model(), eta=1.0, num_data=1)

    X = np.array([[0.0], [0.5], [1.0]])

    assert small.normalized(X) == pytest.approx(large.normalized(X))
    assert _mass(small.normalized(GRID)) == pytest.approx(1.0, rel=1e-3)


def test_discrete_hyperparameters_are_summed_over_their_values():
    """A categorical or integer belief has its mass summed over its values, which ConfigSpace gives density at.

    Setup: a weighted categorical and a normal integer, whose ConfigSpace probabilities sum to about 0.9.
    Expect: the normalized probabilities over the values sum to one.
    """
    categorical = ConfigurationSpace({"k": Categorical("k", ["a", "b", "c"], weights=[6, 3, 1])})
    integer = ConfigurationSpace({"n": Integer("n", (1, 10), distribution=Normal(4, 2))})
    hyperparameter = dict(integer)["n"]

    for space, values in (
        (categorical, np.arange(3.0)),
        (integer, hyperparameter.to_vector(np.arange(1, 11))),
    ):
        weight = PriorWeight(ConfigSpacePrior(space), PolynomialDecay(beta=1.0), t0=0, floor=0.0)
        weight.update(model=Model(), eta=1.0, num_data=2)

        assert weight.normalized(values.reshape(-1, 1)).sum() == pytest.approx(1.0)


def test_the_log_form_is_the_logarithm_of_the_linear_one():
    """The normalized log form agrees with the logarithm of the normalized linear form, floor included."""
    weight = _weight(_space(0.05), 2, floor=1e-6)
    X = np.linspace(0.0, 1.0, 11).reshape(-1, 1)

    assert weight.normalized(X, log=True) == pytest.approx(np.log(weight.normalized(X)))
