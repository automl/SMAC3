from __future__ import annotations

import tempfile

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac import BlackBoxFacade, Scenario
from smac.acquisition.weight import (
    AcceptAllPriors,
    ConfigSpacePrior,
    IncumbentComparisonPolicy,
    PriorWeight,
)
from smac.runhistory import TrialValue
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
