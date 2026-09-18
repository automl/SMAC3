from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float, Normal

from smac.acquisition.weight import (
    ConfigSpacePrior,
    PolynomialDecay,
    PriorEnsemble,
    PriorWeight,
    discretize_pdf,
)
from smac.acquisition.weight.prior import density_of

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class Model:
    def __init__(self, configspace=None):
        self._configspace = configspace

    @property
    def meta(self):
        return {"name": "Model"}


@pytest.fixture
def search_space() -> ConfigurationSpace:
    return ConfigurationSpace({"a": Float("a", (0.0, 1.0)), "b": Float("b", (0.0, 1.0))})


@pytest.fixture
def prior_space() -> ConfigurationSpace:
    """A belief about `a` only; `b` stays uniform and contributes a constant factor."""
    return ConfigurationSpace(
        {"a": Float("a", (0.0, 1.0), distribution=Normal(0.25, 0.1)), "b": Float("b", (0.0, 1.0))}
    )


@pytest.fixture
def X() -> np.ndarray:
    return np.array([[0.25, 0.1], [0.25, 0.9], [0.9, 0.5]])


def _updated(weight, num_data=0, **kwargs):
    weight.update(model=kwargs.pop("model", Model()), eta=1.0, num_data=num_data, **kwargs)

    return weight


def test_the_density_is_the_product_over_hyperparameters(search_space, prior_space, X):
    prior = ConfigSpacePrior(prior_space)
    hyperparameters = list(dict(prior_space).values())

    expected = np.ones((len(X), 1))
    for hyperparameter, column in zip(hyperparameters, X.T):
        expected = expected * density_of(hyperparameter, column)

    assert prior.pdf(X) == pytest.approx(expected)
    assert prior.pdf(X).shape == (len(X), 1)


def test_a_hyperparameter_without_a_belief_contributes_a_constant(prior_space, X):
    """That is how a prior over part of the search space is written: leave the rest uniform."""
    prior = ConfigSpacePrior(prior_space)

    # The first two rows differ only in `b`, which the prior has no opinion about.
    values = prior.pdf(X)

    assert values[0, 0] == pytest.approx(values[1, 0])


def test_a_prior_can_be_sampled_from(prior_space):
    prior = ConfigSpacePrior(prior_space)

    assert len(prior.sample(1)) == 1
    assert len(prior.sample(5)) == 5


def test_a_mismatched_prior_is_rejected(search_space):
    """Evaluating column by column means a misordered prior silently applies each belief to the wrong column."""
    reordered = ConfigurationSpace({"a": Float("a", (0.0, 1.0)), "c": Float("c", (0.0, 1.0))})
    prior = ConfigSpacePrior(reordered)

    with pytest.raises(ValueError, match="match in name and order"):
        prior.validate_against(search_space)

    ConfigSpacePrior(search_space).validate_against(search_space)


def test_the_belief_is_read_off_the_model_when_none_is_given(prior_space, X):
    weight = PriorWeight()

    assert weight.is_active() is False

    _updated(weight, model=Model(prior_space))

    assert weight.is_active() is True
    assert weight(X) == pytest.approx(ConfigSpacePrior(prior_space).pdf(X) + 1e-12)


def test_the_floor_keeps_an_impossible_configuration_reachable(search_space):
    """Without the floor the search could never recover from a mistaken belief."""
    space = ConfigurationSpace({"a": Float("a", (0.0, 1.0), distribution=Normal(0.5, 0.01))})
    weight = _updated(PriorWeight(ConfigSpacePrior(space), floor=1e-6))

    assert weight(np.array([[0.0]]))[0, 0] > 0


def test_the_decay_is_measured_from_the_anchor(prior_space):
    weight = PriorWeight(ConfigSpacePrior(prior_space), t0=20)

    _updated(weight, num_data=20)
    assert weight.steps == 0

    _updated(weight, num_data=35)
    assert weight.steps == 15


def test_the_anchor_latches_at_the_first_update_when_not_given(prior_space):
    weight = PriorWeight(ConfigSpacePrior(prior_space))

    assert weight.t0 is None

    _updated(weight, num_data=12)

    assert weight.t0 == 12
    assert weight.steps == 0


def test_an_anchored_prior_refuses_to_be_re_anchored(prior_space):
    """Re-anchoring silently restores full strength, so the belief would never fade."""
    weight = PriorWeight(ConfigSpacePrior(prior_space), t0=20)

    with pytest.raises(ValueError, match="already anchored"):
        weight.anchor(40)


def test_a_later_prior_is_stronger_than_an_older_one_at_the_same_trial(prior_space, X):
    """The mechanism of DynaBO: a belief supplied at trial 120 arrives at full strength.

    Both weights carry the same belief and the same decay factor, so any difference between them at the same trial
    count comes only from when they were supplied.
    """
    decay = PolynomialDecay(beta=20.0)
    early = _updated(PriorWeight(ConfigSpacePrior(prior_space), decay, t0=10), num_data=100)
    late = _updated(PriorWeight(ConfigSpacePrior(prior_space), decay, t0=100), num_data=100)

    assert early.steps == 90
    assert late.steps == 0
    assert early.decay(early.steps) < late.decay(late.steps)

    # The older belief has been flattened towards one; the newer one still has an opinion.
    spread = lambda values: values.max() - values.min()  # noqa: E731
    assert spread(early(X)) < spread(late(X))


def test_the_density_is_coarsened_for_a_random_forest(prior_space):
    """A random forest needs a piecewise constant acquisition function to stay well behaved."""
    space = ConfigurationSpace({"a": Float("a", (0.0, 1.0), distribution=Normal(0.5, 0.2))})
    X = np.linspace(0, 1, 500).reshape((-1, 1))

    exact = _updated(PriorWeight(ConfigSpacePrior(space), PolynomialDecay(beta=2.0)))
    coarse = _updated(PriorWeight(ConfigSpacePrior(space), PolynomialDecay(beta=2.0), discretize=True))

    # The exact density is symmetric, so it has fewer distinct values than points, but far more than the bins.
    assert len(np.unique(coarse(X))) <= int(np.ceil(10.0 * 2.0))
    assert len(np.unique(exact(X))) > len(np.unique(coarse(X)))


def test_the_coarsening_follows_the_decay(prior_space):
    """A belief which has faded is also flattened, so it stops fighting the surrogate for resolution."""
    space = ConfigurationSpace({"a": Float("a", (0.0, 1.0), distribution=Normal(0.5, 0.2))})
    X = np.linspace(0, 1, 500).reshape((-1, 1))

    weight = PriorWeight(ConfigSpacePrior(space), PolynomialDecay(beta=40.0), discretize=True, t0=0)

    _updated(weight, num_data=0)
    early = len(np.unique(weight(X)))

    _updated(weight, num_data=40)
    late = len(np.unique(weight(X)))

    assert late < early


def test_coarsening_needs_at_least_one_bin():
    space = ConfigurationSpace({"a": Float("a", (0.0, 1.0), distribution=Normal(0.5, 0.2))})
    hyperparameter = dict(space)["a"]

    with pytest.raises(ValueError, match="at least one"):
        discretize_pdf(hyperparameter, np.linspace(0, 1, 10), number_of_bins=0)


def test_priors_are_summed_by_default(prior_space, X):
    """Summing reads as 'any of these regions is worth a look'; multiplying would let them cancel out."""
    ensemble = PriorEnsemble()

    assert ensemble.combination == "sum"

    first = PriorWeight(ConfigSpacePrior(prior_space), t0=0)
    second = PriorWeight(ConfigSpacePrior(prior_space), t0=0)
    ensemble.add(first, key="first")
    ensemble.add(second, key="second")
    _updated(ensemble, num_data=0)

    assert ensemble(X) == pytest.approx(first(X) + second(X))


def test_a_faded_prior_can_be_pruned(prior_space):
    ensemble = PriorEnsemble(prune_exponent=0.5)
    ensemble.add(PriorWeight(ConfigSpacePrior(prior_space), PolynomialDecay(beta=2.0), t0=0), key="old")

    _updated(ensemble, num_data=0)
    assert "old" in ensemble.members

    _updated(ensemble, num_data=10)
    assert "old" not in ensemble.members


def test_meta_describes_the_prior(prior_space):
    weight = PriorWeight(ConfigSpacePrior(prior_space), PolynomialDecay(beta=5.0), t0=7, name="from the user")
    meta = weight.meta

    assert meta["name"] == "PriorWeight"
    assert meta["t0"] == 7
    assert meta["decay"]["beta"] == 5.0
    assert meta["prior"]["hyperparameters"] == ["a", "b"]
