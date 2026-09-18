from __future__ import annotations

import tempfile

import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac import BlackBoxFacade, Scenario
from smac.acquisition.function import WeightedAcquisitionFunction
from smac.acquisition.weight import ConfigSpacePrior, PolynomialDecay, PriorEnsemble, PriorWeight
from smac.callback import Callback
from smac.runhistory import TrialValue
from smac.utils.configspace import (
    convert_configurations_to_array,
    create_prior_configspace_copy,
)

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


OPTIMUM = 0.85


def _configspace() -> ConfigurationSpace:
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Float("x0", (0.0, 1.0)), Float("x1", (0.0, 1.0)))

    return configspace


def target(config, seed: int = 0):
    """A bowl whose optimum sits off centre, so a belief pointing at it is worth something."""
    return (config["x0"] - OPTIMUM) ** 2 + (config["x1"] - OPTIMUM) ** 2


def _smac(n_trials: int = 40, seed: int = 0, **kwargs):
    scenario = Scenario(
        _configspace(),
        n_trials=n_trials,
        seed=seed,
        deterministic=True,
        output_directory=tempfile.mkdtemp(),
    )

    return BlackBoxFacade(scenario, target, overwrite=True, logging_level=60, **kwargs)


def _warm_up(smac, n: int = 8):
    for _ in range(n):
        info = smac.ask()
        smac.tell(info, TrialValue(cost=target(info.config), time=0.1))


def test_a_belief_can_be_stated_while_the_run_is_going_on():
    smac = _smac()
    _warm_up(smac)

    key = smac.add_prior({"x0": OPTIMUM, "x1": OPTIMUM})

    assert key in smac.priors
    assert isinstance(smac.optimizer.config_selector._acquisition_function, WeightedAcquisitionFunction)
    assert isinstance(smac.optimizer.config_selector.priors_ensemble, PriorEnsemble)


def test_a_belief_is_anchored_where_it_was_stated():
    """A belief stated at trial 20 arrives at full strength, not at the exponent an older one decayed to."""
    smac = _smac()
    _warm_up(smac)

    expected = smac.optimizer.config_selector._current_num_data()
    key = smac.add_prior({"x0": OPTIMUM})

    assert smac.priors[key].t0 == expected


def test_two_beliefs_stated_at_different_times_carry_different_influence():
    smac = _smac(n_trials=60)
    _warm_up(smac)

    early = smac.add_prior({"x0": OPTIMUM}, key="early")
    _warm_up(smac, 10)
    late = smac.add_prior({"x0": OPTIMUM}, key="late")

    # Bring both up to date under the same trial count.
    smac.ask()

    assert smac.priors[early].steps > smac.priors[late].steps
    assert smac.priors[early].decay(smac.priors[early].steps) < smac.priors[late].decay(smac.priors[late].steps)


def test_the_influence_of_a_belief_decays():
    """Frozen model, same belief, later trial: the belief has to say less about where to look."""
    configspace = _configspace()
    prior = ConfigSpacePrior(create_prior_configspace_copy(configspace, {"x0": OPTIMUM, "x1": OPTIMUM}))
    weight = PriorWeight(prior, PolynomialDecay(beta=10.0), t0=0)

    X = convert_configurations_to_array(list(configspace.sample_configuration(200)))

    weight.update(model=None, eta=1.0, num_data=1)
    early = weight(X)

    weight.update(model=None, eta=1.0, num_data=100)
    late = weight(X)

    assert (late.max() / late.min()) < (early.max() / early.min())


def test_a_belief_can_be_withdrawn():
    smac = _smac()
    _warm_up(smac)

    key = smac.add_prior({"x0": OPTIMUM})
    assert key in smac.priors

    smac.remove_prior(key)

    assert key not in smac.priors
    assert smac.optimizer.config_selector.priors == {}


def test_the_maximizer_draws_candidates_from_the_belief():
    """Weighting alone is not enough: a sharply peaked belief has to be sampled from to be reachable."""
    smac = _smac()
    _warm_up(smac)

    key = smac.add_prior({"x0": OPTIMUM, "x1": OPTIMUM})
    pool = smac.optimizer.config_selector._acquisition_maximizer._random_search.sampling_pool

    assert key in pool.sources


def test_a_belief_stated_from_a_callback_reaches_the_very_next_configuration():
    """The regression test for the challengers already ranked when the belief arrived."""

    class StateBelief(Callback):
        def __init__(self):
            self.added_at = None
            self.maximizations = 0

        def on_ask_start(self, smbo):
            if self.added_at is None and smbo.runhistory.finished >= 8:
                self.added_at = smbo.runhistory.finished
                smbo.add_prior({"x0": OPTIMUM, "x1": OPTIMUM}, key="from the callback")

    callback = StateBelief()
    smac = _smac(callbacks=[callback])

    for _ in range(12):
        info = smac.ask()
        smac.tell(info, TrialValue(cost=target(info.config), time=0.1))

    assert callback.added_at is not None

    selector = smac.optimizer.config_selector
    # The acquisition function was brought up to date, and the stale challengers discarded, on that same ask.
    assert "from the callback" in selector.priors
    assert selector._acquisition_needs_update is False
    assert selector._force_retrain is False


def test_callbacks_are_told_about_a_belief():
    class Record(Callback):
        def __init__(self):
            self.added = []
            self.removed = []

        def on_prior_added(self, smbo, key, prior):
            self.added.append(key)

        def on_prior_removed(self, smbo, key, prior):
            self.removed.append(key)

    record = Record()
    smac = _smac(callbacks=[record])
    _warm_up(smac)

    key = smac.add_prior({"x0": OPTIMUM})
    smac.remove_prior(key)

    assert record.added == [key]
    assert record.removed == [key]


def test_a_belief_pointing_at_the_optimum_helps():
    """The reason for all of this: a correct belief stated part way through should pay off."""

    def run(with_prior: bool) -> float:
        smac = _smac(n_trials=45, seed=1)

        for i in range(45):
            if with_prior and i == 12:
                smac.add_prior({"x0": OPTIMUM, "x1": OPTIMUM})

            info = smac.ask()
            smac.tell(info, TrialValue(cost=target(info.config), time=0.1))

        return min(value.cost for value in smac.runhistory._data.values())

    assert run(with_prior=True) <= run(with_prior=False)


def test_a_belief_about_something_the_search_space_does_not_have_is_rejected():
    smac = _smac()
    _warm_up(smac)

    with pytest.raises(ValueError, match="not in the search space"):
        smac.add_prior({"nope": 0.5})


def test_a_prior_weight_carries_its_own_decay():
    smac = _smac()
    _warm_up(smac)

    configspace = create_prior_configspace_copy(smac.scenario.configspace, {"x0": OPTIMUM})
    weight = PriorWeight(ConfigSpacePrior(configspace), PolynomialDecay(beta=3.0))

    with pytest.raises(ValueError, match="carries its own decay"):
        smac.add_prior(weight, decay=PolynomialDecay(beta=1.0))

    key = smac.add_prior(weight)

    assert smac.priors[key] is weight
