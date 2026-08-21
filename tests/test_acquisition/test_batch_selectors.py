from __future__ import annotations

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float

from smac.acquisition.batch_selector import (
    AbstractBatchSelector,
    StochasticBatchSelector,
)
from smac.acquisition.function import EI
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.model.random_forest.random_forest import RandomForest

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


@pytest.fixture
def configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=0)
    cs.add([Float("x", (0.0, 1.0)), Float("y", (0.0, 1.0))])

    return cs


@pytest.fixture
def candidates(configspace) -> list:
    return configspace.sample_configuration(20)


@pytest.fixture
def acquisition_values() -> np.ndarray:
    # Descending, as the maximizer hands them over.
    return np.linspace(10.0, 1.0, 20)


class _JointSampleSelector(AbstractBatchSelector):
    """Minimal selector that only exists to exercise the capability checks."""

    @property
    def requires_joint_samples(self) -> bool:
        return True

    def _select(self, candidates, acquisition_values, batch_size, model, acquisition_function, pending):
        return list(candidates)


class _ConditioningSelector(AbstractBatchSelector):
    """Minimal selector that only exists to exercise the capability checks."""

    @property
    def requires_conditioning(self) -> bool:
        return True

    def _select(self, candidates, acquisition_values, batch_size, model, acquisition_function, pending):
        return list(candidates)


def _selection_order(selected: list, candidates: list) -> list[int]:
    """Maps the selected configurations back to their index in the candidate list."""
    lookup = {config: i for i, config in enumerate(candidates)}

    return [lookup[config] for config in selected]


def test_select_rejects_mismatched_lengths(candidates, acquisition_values):
    """Checks that a candidate list and a value array of different length is refused.

    Passes one fewer acquisition value than there are candidates and asserts the ValueError, since
    silently zipping them would misattribute scores to configurations.
    """
    selector = StochasticBatchSelector(seed=0)

    with pytest.raises(ValueError, match="acquisition values"):
        selector.select(candidates, acquisition_values[:-1], batch_size=4)


def test_select_returns_every_candidate(candidates, acquisition_values):
    """Checks that selection reorders rather than truncates the candidate list.

    Selects a batch of four out of twenty and asserts the returned list is a permutation of the
    input, which is what lets the caller keep drawing configurations past the batch.
    """
    selector = StochasticBatchSelector(seed=0)
    selected = selector.select(candidates, acquisition_values, batch_size=4)

    assert len(selected) == len(candidates)
    assert sorted(_selection_order(selected, candidates)) == list(range(len(candidates)))


@pytest.mark.parametrize("batch_size", [0, 1])
def test_select_passes_through_trivial_batches(candidates, acquisition_values, batch_size):
    """Checks that a batch of at most one is handed back untouched.

    Calls select with batch sizes 0 and 1 and asserts the acquisition order survives, because there
    is no batch to trade off against when only one configuration is wanted.
    """
    selector = StochasticBatchSelector(seed=0)
    selected = selector.select(candidates, acquisition_values, batch_size=batch_size)

    assert selected == list(candidates)


def test_select_passes_through_single_candidate(candidates, acquisition_values):
    """Checks that a one-element candidate pool is handed back untouched.

    Calls select with a single candidate and asserts it is returned as-is, since there is nothing
    to reorder.
    """
    selector = StochasticBatchSelector(seed=0)
    selected = selector.select(candidates[:1], acquisition_values[:1], batch_size=8)

    assert selected == candidates[:1]


def test_joint_sample_requirement_is_enforced(configspace, candidates, acquisition_values):
    """Checks that a selector needing joint samples refuses a model that cannot provide them.

    Passes no model at all and asserts the error names the capability, so the failure is loud rather
    than a batch built on meaningless correlations.
    """
    selector = _JointSampleSelector(seed=0)

    with pytest.raises(ValueError, match="supports_joint_samples"):
        selector.select(candidates, acquisition_values, batch_size=4, model=None)


def test_conditioning_requirement_is_enforced(configspace, candidates, acquisition_values):
    """Checks that a selector needing conditioning refuses a random forest.

    Trains a forest, which cannot be conditioned in closed form, and asserts the error explains that
    such models would produce identical batch members.
    """
    model = RandomForest(configspace=configspace, seed=0)
    model.train(np.random.RandomState(0).rand(10, 2), np.random.RandomState(0).rand(10, 1))
    selector = _ConditioningSelector(seed=0)

    with pytest.raises(ValueError, match="supports_conditioning"):
        selector.select(candidates, acquisition_values, batch_size=4, model=model)


def test_stochastic_rejects_invalid_arguments():
    """Checks that the constructor validates its knobs.

    Passes an unknown mode, a negative exponent and a non-positive temperature and asserts each is
    refused up front rather than producing a silently degenerate distribution.
    """
    with pytest.raises(ValueError, match="mode"):
        StochasticBatchSelector(mode="nonsense")

    with pytest.raises(ValueError, match="alpha"):
        StochasticBatchSelector(alpha=-1.0)

    with pytest.raises(ValueError, match="temperature"):
        StochasticBatchSelector(temperature=0.0)


def test_stochastic_is_deterministic_for_a_seed(candidates, acquisition_values):
    """Checks that two selectors with the same seed pick the same batch.

    Runs selection twice from freshly seeded selectors and asserts the orders match, so that a SMAC
    run stays reproducible.
    """
    first = StochasticBatchSelector(seed=7).select(candidates, acquisition_values, batch_size=8)
    second = StochasticBatchSelector(seed=7).select(candidates, acquisition_values, batch_size=8)

    assert first == second


def test_stochastic_differs_from_top_k(candidates, acquisition_values):
    """Checks that the selector actually deviates from taking the highest scoring candidates.

    Compares the first eight selected against the first eight by acquisition value and asserts they
    are not identical, which is the entire point of sampling rather than sorting.
    """
    selector = StochasticBatchSelector(seed=0)
    selected = selector.select(candidates, acquisition_values, batch_size=8)

    assert _selection_order(selected, candidates)[:8] != list(range(8))


@pytest.mark.parametrize("mode", ["soft_rank", "power", "softmax"])
def test_stochastic_concentrates_as_randomness_vanishes(candidates, acquisition_values, mode):
    """Checks that every mode recovers the plain acquisition order in its deterministic limit.

    Drives the exponent very high (or the temperature very low) so the weights dominate the Gumbel
    noise, and asserts the result is exactly the descending acquisition order.
    """
    selector = StochasticBatchSelector(mode=mode, alpha=500.0, temperature=1e-6, seed=0)
    selected = selector.select(candidates, acquisition_values, batch_size=8)

    assert _selection_order(selected, candidates) == list(range(len(candidates)))


def test_stochastic_becomes_uniform_without_weighting(candidates, acquisition_values):
    """Checks that an exponent of zero turns the selection into a uniform shuffle.

    Draws the top candidate 400 times with alpha zero and asserts the mean index is close to the
    middle of the pool, confirming the acquisition values no longer bias the draw.
    """
    picks = [
        _selection_order(
            StochasticBatchSelector(mode="soft_rank", alpha=0.0, seed=seed).select(
                candidates, acquisition_values, batch_size=8
            ),
            candidates,
        )[0]
        for seed in range(400)
    ]

    assert abs(np.mean(picks) - (len(candidates) - 1) / 2) < 2.0


def test_stochastic_prefers_better_candidates(candidates, acquisition_values):
    """Checks that the default weighting still favours high acquisition values.

    Draws the top candidate 400 times and asserts the mean index sits well inside the better half of
    the pool, so the added randomness does not throw away the surrogate's ranking.
    """
    uniform_mean = (len(candidates) - 1) / 2
    picks = [
        _selection_order(
            StochasticBatchSelector(seed=seed).select(candidates, acquisition_values, batch_size=8),
            candidates,
        )[0]
        for seed in range(400)
    ]

    assert np.mean(picks) < uniform_mean / 2


def test_stochastic_skips_pending_configurations(candidates, acquisition_values):
    """Checks that configurations already being evaluated are not proposed again.

    Marks the five best candidates as pending and asserts none of them appear in the result, since
    they are already part of the batch being formed.
    """
    pending = list(candidates[:5])
    selector = StochasticBatchSelector(seed=0)
    selected = selector.select(candidates, acquisition_values, batch_size=8, pending=pending)

    assert len(selected) == len(candidates) - len(pending)
    assert all(config not in pending for config in selected)


def test_maximizer_without_selector_keeps_acquisition_order(configspace):
    """Checks that the maximizer is unchanged when no batch selector is configured.

    Runs the default maximizer twice with the same seed and asserts the challengers match, guarding
    the default path against the new hook.
    """
    model = RandomForest(configspace=configspace, seed=0)
    rng = np.random.RandomState(0)
    model.train(rng.rand(20, 2), rng.rand(20, 1))

    acquisition_function = EI()
    acquisition_function.update(model=model, eta=0.5)

    def challengers(**kwargs):
        # The configuration space carries its own random state, which the random search advances.
        configspace.seed(0)
        maximizer = LocalAndSortedRandomSearch(
            configspace=configspace,
            acquisition_function=acquisition_function,
            challengers=100,
            seed=0,
            **kwargs,
        )
        return list(maximizer.maximize([], random_design=None))

    assert challengers() == challengers()


def test_maximizer_applies_the_batch_selector(configspace):
    """Checks that the maximizer hands its candidates to the batch selector.

    Compares the challengers produced with and without a stochastic selector and asserts they differ,
    and that attaching a selector widens the candidate pool beyond the local search results.
    """
    model = RandomForest(configspace=configspace, seed=0)
    rng = np.random.RandomState(0)
    model.train(rng.rand(20, 2), rng.rand(20, 1))

    acquisition_function = EI()
    acquisition_function.update(model=model, eta=0.5)

    def challengers(selector):
        configspace.seed(0)
        maximizer = LocalAndSortedRandomSearch(
            configspace=configspace,
            acquisition_function=acquisition_function,
            challengers=100,
            seed=0,
            batch_selector=selector,
            batch_pool_size=64,
        )
        return list(maximizer.maximize([], random_design=None, batch_size=8))

    baseline = challengers(None)
    stochastic = challengers(StochasticBatchSelector(seed=0))

    assert len(stochastic) > len(baseline)
    assert stochastic[:8] != baseline[:8]


def test_pending_trials_reach_the_batch_selector(configspace):
    """Checks that configurations handed out but not yet evaluated are passed down as pending.

    Drives a facade with ask() eight times without telling anything back, records what the selector
    was given, and asserts the pending list matches the trials the run history reports as running.
    This is the plumbing that lets a selector avoid re-proposing work that is already in flight.
    """
    from smac import HyperparameterOptimizationFacade, Scenario
    from smac.runhistory import TrialValue

    def target(config, seed: int = 0) -> float:
        return float(config["x"] ** 2 + config["y"] ** 2)

    seen_pending: list[list] = []

    class RecordingSelector(StochasticBatchSelector):
        def _select(self, candidates, acquisition_values, batch_size, model, acquisition_function, pending):
            seen_pending.append(list(pending))
            return super()._select(
                candidates, acquisition_values, batch_size, model, acquisition_function, pending
            )

    scenario = Scenario(configspace, n_trials=40, seed=0, deterministic=True)
    smac = HyperparameterOptimizationFacade(
        scenario,
        target,
        batch_selector=RecordingSelector(seed=0),
        overwrite=True,
        logging_level=40,
    )

    for _ in range(12):
        info = smac.ask()
        smac.tell(info, TrialValue(cost=target(info.config)))

    asked = [smac.ask().config for _ in range(8)]
    running = smac.runhistory.get_running_configs()

    assert len(set(asked)) == len(asked)
    assert all(config in running for config in asked)
    assert seen_pending, "the batch selector was never called"
    assert any(len(pending) > 0 for pending in seen_pending)
