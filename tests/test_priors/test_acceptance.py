from __future__ import annotations

import tempfile

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac import BlackBoxFacade, Scenario
from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.weight import (
    AbstractPriorAcceptancePolicy,
    AcceptAllPriors,
    ClimbingComparisonPolicy,
    ConfigSpacePrior,
    IncumbentComparisonPolicy,
    PriorWeight,
    TabulatedPrior,
)
from smac.acquisition.weight.acceptance import _BeliefWeightedBound, _in_objective_units
from smac.runhistory import TrialInfo, TrialValue
from smac.runhistory.encoder import RunHistoryEIPSEncoder, RunHistoryLogEncoder
from smac.utils.configspace import create_prior_configspace_copy

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


OPTIMUM = 0.85


def _configspace() -> ConfigurationSpace:
    configspace = ConfigurationSpace(seed=0)
    configspace.add(Float("x0", (0.0, 1.0)), Float("x1", (0.0, 1.0)))

    return configspace


def target(config, seed: int = 0):
    return (config["x0"] - OPTIMUM) ** 2 + (config["x1"] - OPTIMUM) ** 2


def _smac(n_trials: int = 40, **kwargs):
    scenario = Scenario(
        _configspace(),
        n_trials=n_trials,
        seed=0,
        deterministic=True,
        output_directory=tempfile.mkdtemp(),
    )

    return BlackBoxFacade(scenario, target, overwrite=True, logging_level=60, **kwargs)


def _warm_up(smac, n: int = 20):
    for _ in range(n):
        info = smac.ask()
        smac.tell(info, TrialValue(cost=target(info.config), time=0.1))


def _prior(configspace, values) -> PriorWeight:
    return PriorWeight(ConfigSpacePrior(create_prior_configspace_copy(configspace, values, std_denominator=20.0)))


def _context(smac):
    selector = smac.optimizer.config_selector

    return {
        "configspace": smac.scenario.configspace,
        "model": selector._model,
        "runhistory": smac.runhistory,
        "incumbent": selector._incumbent(),
        "rng": np.random.RandomState(0),
    }


def test_the_default_policy_accepts_everything():
    policy = AcceptAllPriors()

    assert policy.accept(_prior(_configspace(), {"x0": 0.0})) is True


def test_a_belief_pointing_at_a_good_region_is_accepted():
    smac = _smac()
    _warm_up(smac)

    policy = IncumbentComparisonPolicy()

    assert policy.accept(_prior(smac.scenario.configspace, {"x0": OPTIMUM, "x1": OPTIMUM}), **_context(smac)) is True


def test_a_belief_pointing_at_a_bad_region_is_rejected():
    smac = _smac()
    _warm_up(smac)

    policy = IncumbentComparisonPolicy()
    far_from_the_optimum = _prior(smac.scenario.configspace, {"x0": 0.0, "x1": 0.0})

    assert policy.accept(far_from_the_optimum, **_context(smac)) is False


def test_nothing_is_rejected_before_anything_is_known():
    """Otherwise exactly the beliefs stated at the start of a run would be rejected."""
    smac = _smac()
    policy = IncumbentComparisonPolicy()

    assert (
        policy.accept(
            _prior(smac.scenario.configspace, {"x0": 0.0}),
            configspace=smac.scenario.configspace,
            model=None,
            runhistory=smac.runhistory,
            incumbent=None,
        )
        is True
    )


def test_a_belief_stated_before_the_model_is_fitted_is_accepted():
    """The model object exists from the start; asking an unfitted one to predict raises.

    This is the case a run actually hits - a belief stated at the end of the initial design, before the
    surrogate has been trained even once.
    """
    smac = _smac()
    selector = smac.optimizer.config_selector

    assert selector._model is not None, "the model object exists from the start"
    assert selector._trained_model() is None, "but it has not been fitted to anything"

    key = smac.add_prior({"x0": 0.0, "x1": 0.0}, acceptance_policy=IncumbentComparisonPolicy())

    assert key is not None


def test_the_threshold_decides():
    smac = _smac()
    _warm_up(smac)

    bad = _prior(smac.scenario.configspace, {"x0": 0.0, "x1": 0.0})

    assert IncumbentComparisonPolicy(threshold=-1e9).accept(bad, **_context(smac)) is True
    assert IncumbentComparisonPolicy(threshold=1e9).accept(bad, **_context(smac)) is False


def test_a_rejected_belief_changes_nothing():
    smac = _smac()
    _warm_up(smac)

    key = smac.add_prior({"x0": 0.0, "x1": 0.0}, acceptance_policy=IncumbentComparisonPolicy(threshold=1e9))

    assert key is None
    assert smac.priors == {}
    assert smac.optimizer.config_selector.priors_ensemble is None


def test_an_accepted_belief_goes_through():
    smac = _smac()
    _warm_up(smac)

    key = smac.add_prior(
        {"x0": OPTIMUM, "x1": OPTIMUM},
        acceptance_policy=IncumbentComparisonPolicy(threshold=-1e9),
    )

    assert key in smac.priors


def test_the_policy_can_be_set_for_the_whole_run():
    smac = _smac()
    selector = smac.optimizer.config_selector

    assert isinstance(selector.prior_acceptance_policy, AcceptAllPriors)

    selector.prior_acceptance_policy = IncumbentComparisonPolicy(threshold=1e9)
    _warm_up(smac)

    assert smac.add_prior({"x0": 0.0, "x1": 0.0}) is None


def test_at_least_one_sample_is_needed():
    with pytest.raises(ValueError, match="At least one sample"):
        IncumbentComparisonPolicy(n_samples=0)


def test_meta_describes_the_policy():
    meta = IncumbentComparisonPolicy(n_samples=50, threshold=-0.2).meta

    assert meta["name"] == "IncumbentComparisonPolicy"
    assert meta["n_samples"] == 50
    assert meta["threshold"] == -0.2
    assert meta["acquisition_function"]["name"] == "LCB"



# Judging where a belief leads: `ClimbingComparisonPolicy`


def _space(n: int) -> ConfigurationSpace:
    configspace = ConfigurationSpace(seed=0)
    configspace.add([Float(f"x{i}", (0.0, 1.0)) for i in range(n)])

    return configspace


def wide_target(config, seed: int = 0):
    """Every hyperparameter matters, and each is best at `OPTIMUM`."""
    return sum((config[name] - OPTIMUM) ** 2 for name in config)


def _wide_smac(n_hyperparameters: int = 4, n_trials: int = 30):
    """A run over `n_hyperparameters` hyperparameters, all of which matter, warmed up with `n_trials` trials."""
    scenario = Scenario(
        _space(n_hyperparameters),
        n_trials=200,
        seed=0,
        deterministic=True,
        output_directory=tempfile.mkdtemp(),
    )
    smac = BlackBoxFacade(scenario, wide_target, overwrite=True, logging_level=60)
    for _ in range(n_trials):
        info = smac.ask()
        smac.tell(info, TrialValue(cost=wide_target(info.config), time=0.1))

    return smac


def _about_x0(configspace, at: float) -> TabulatedPrior:
    """A belief about `x0` alone, peaked at `at`."""
    positions = np.linspace(0.0, 1.0, 101)

    return TabulatedPrior(configspace, {"x0": list(zip(positions, np.exp(-0.5 * ((positions - at) / 0.05) ** 2)))})


def _judging_context(smac):
    selector = smac.optimizer.config_selector
    model, eta = selector._fit_model_for_judging()

    return {
        "configspace": smac.scenario.configspace,
        "model": model,
        "runhistory": smac.runhistory,
        "incumbent": selector._incumbent(),
        "rng": np.random.RandomState(0),
        "eta": eta,
        "num_data": selector._current_num_data(),
        "runhistory_encoder": selector._runhistory_encoder,
    }


def test_a_correct_belief_about_one_hyperparameter_is_accepted():
    """The case the policy exists for: a belief about one of four hyperparameters, pointing at its optimum. Drawn
    at random, the three it says nothing about drag its mean down far enough that comparing means refuses it;
    judged by where it leads, it is accepted."""
    smac = _wide_smac()

    assert smac.add_prior(_about_x0(smac.scenario.configspace, OPTIMUM), acceptance_policy=IncumbentComparisonPolicy()) is None

    smac = _wide_smac()

    assert smac.add_prior(_about_x0(smac.scenario.configspace, OPTIMUM), acceptance_policy=ClimbingComparisonPolicy()) is not None


def test_a_wrong_belief_about_one_hyperparameter_is_still_rejected():
    smac = _wide_smac()

    key = smac.add_prior(_about_x0(smac.scenario.configspace, 0.05), acceptance_policy=ClimbingComparisonPolicy())

    assert key is None
    assert smac.priors == {}


def test_the_same_random_state_gives_the_same_judgement():
    """Every draw and every climb comes from a copy of the random state handed over, so judging twice from the same
    state gives the same scores - and leaves that state as it was."""
    smac = _wide_smac()
    context = _judging_context(smac)
    rng = context["rng"]
    before = rng.get_state()[1].copy()
    belief = PriorWeight(_about_x0(smac.scenario.configspace, OPTIMUM))
    policy = ClimbingComparisonPolicy(n_samples=500)

    first = policy.compare(belief, **context)
    second = policy.compare(belief, **context)

    assert first == second
    np.testing.assert_array_equal(rng.get_state()[1], before)


def test_judging_does_not_anchor_the_belief():
    """The belief is judged at the strength it would be added with, and is not anchored by being judged - `add_prior`
    anchors it afterwards, and anchoring twice is refused."""
    smac = _wide_smac()
    belief = PriorWeight(_about_x0(smac.scenario.configspace, OPTIMUM))

    ClimbingComparisonPolicy(n_samples=500, threshold=-1e9).accept(belief, **_judging_context(smac))

    assert belief.t0 is None
    assert smac.add_prior(belief) is not None


def test_a_belief_still_weights_where_nothing_beats_the_incumbent():
    """Where even the optimistic estimate does not improve on the incumbent, clipping would score everything zero and
    a climb could not move. The belief keeps ordering those configurations: the one it favours scores higher, and
    every shortfall still scores below every improvement."""

    class Fixed(AbstractAcquisitionFunction):
        """Returns the negated cost bound it is given in each row's second column."""

        def __init__(self) -> None:
            super().__init__()

        def _compute(self, X: np.ndarray) -> np.ndarray:
            return -X[:, 1:2]

    belief = PriorWeight(_about_x0(_space(2), 0.5))
    weighted = _BeliefWeightedBound(Fixed(), belief, eta=1.0)

    # Each row is (x0, the bound there). Bounds of 2.0 fall short of the incumbent's 1.0; the first point sits at the
    # belief's peak, the second far from it.
    short_near, short_far, improving = weighted._compute(np.array([[0.5, 2.0], [0.9, 2.0], [0.9, 0.5]]))[:, 0]

    assert short_near > short_far
    assert improving > short_near


def test_a_policy_that_needs_a_model_is_given_one_fitted_to_everything():
    """Trials reported with `tell` alone never train the model, which only an `ask` does. A policy that asks for a
    model is handed one trained on every trial, with the incumbent's cost, the trial count and the encoder."""

    class Recording(AbstractPriorAcceptancePolicy):
        seen: dict = {}

        @property
        def requires_model(self) -> bool:
            return True

        def accept(self, prior, **kwargs):  # noqa: D102
            Recording.seen = kwargs
            return False

    smac = _smac()
    _warm_up(smac, 12)
    for config in smac.scenario.configspace.sample_configuration(5):
        smac.tell(TrialInfo(config=config, seed=0), TrialValue(cost=target(config), time=0.1))

    smac.add_prior({"x0": OPTIMUM}, acceptance_policy=Recording())
    seen = Recording.seen

    assert seen["model"] is not None
    assert seen["num_data"] == len(smac.runhistory)
    assert seen["eta"] is not None
    assert seen["runhistory_encoder"] is smac.optimizer.config_selector._runhistory_encoder


def test_a_policy_that_needs_no_model_gets_the_arguments_it_always_had():
    """A policy that does not ask for a model is called as before - nothing is trained for it, and it is not handed
    the arguments it was never written to take."""

    class Strict(AbstractPriorAcceptancePolicy):
        def accept(self, prior, *, configspace, model, runhistory, incumbent, rng=None):  # noqa: D102
            return True

    smac = _smac()
    _warm_up(smac, 12)

    assert smac.add_prior({"x0": OPTIMUM}, acceptance_policy=Strict()) is not None


def test_the_sample_count_can_follow_the_number_of_hyperparameters():
    """`n_samples_per_hyperparameter` overrides the fixed count with that many per hyperparameter."""
    smac = _wide_smac(n_hyperparameters=4, n_trials=12)
    drawn = []
    belief = _about_x0(smac.scenario.configspace, OPTIMUM)
    sample = belief.sample

    def recording_sample(n, rng=None):
        drawn.append(n)
        return sample(n, rng)

    belief.sample = recording_sample  # type: ignore[method-assign]

    IncumbentComparisonPolicy(n_samples_per_hyperparameter=7).accept(PriorWeight(belief), **_judging_context(smac))

    assert drawn == [28]


def test_scores_are_read_in_the_objective_units():
    """A confidence bound is the negated bound on the cost as the model sees it, so it is mapped back through the
    encoder; an encoder that cannot be inverted leaves the scores as they are."""
    scores = np.array([[-1.0], [-2.0]])
    log = RunHistoryLogEncoder(scenario=_smac().scenario)
    eips = RunHistoryEIPSEncoder(scenario=_smac().scenario)

    np.testing.assert_allclose(_in_objective_units(scores, log), -np.exp([[1.0], [2.0]]))
    np.testing.assert_allclose(_in_objective_units(scores, eips), scores)
    np.testing.assert_allclose(_in_objective_units(scores, None), scores)


def test_the_climbing_policy_checks_its_arguments():
    with pytest.raises(ValueError, match="At least one climb"):
        ClimbingComparisonPolicy(top_k=0)

    with pytest.raises(ValueError, match="neighbourhood share"):
        ClimbingComparisonPolicy(neighbourhood_share=1.5)

    with pytest.raises(ValueError, match="per hyperparameter"):
        ClimbingComparisonPolicy(n_samples_per_hyperparameter=0)


def test_meta_describes_the_climbing_policy():
    meta = ClimbingComparisonPolicy(top_k=5, neighbourhood_share=0.25, n_samples_per_hyperparameter=100).meta

    assert meta["name"] == "ClimbingComparisonPolicy"
    assert meta["top_k"] == 5
    assert meta["neighbourhood_share"] == 0.25
    assert meta["n_samples_per_hyperparameter"] == 100
    assert meta["n_samples"] == 5000
