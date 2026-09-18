from __future__ import annotations

import pytest
from ConfigSpace import ConfigurationSpace, Float, Normal

from smac.acquisition.function import EI
from smac.acquisition.maximizer import LocalAndSortedRandomSearch, RandomSearch
from smac.acquisition.maximizer.local_search import LocalSearch
from smac.acquisition.maximizer.sampling import SamplingPool

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


@pytest.fixture
def search_space() -> ConfigurationSpace:
    return ConfigurationSpace({"a": Float("a", (0.0, 1.0))}, seed=0)


@pytest.fixture
def prior_space() -> ConfigurationSpace:
    return ConfigurationSpace({"a": Float("a", (0.0, 1.0), distribution=Normal(0.9, 0.01))}, seed=0)


class Acquisition(EI):
    def _update(self, **kwargs):
        pass

    def _compute(self, X):
        return X[:, 0].reshape((-1, 1))


def test_the_search_space_gets_everything_on_its_own(search_space):
    assert SamplingPool(search_space).counts(10) == {"default": 10}


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 10, 999])
def test_the_split_always_adds_up(search_space, prior_space, n):
    pool = SamplingPool(search_space)
    pool.add("one", prior_space, weight=0.3)
    pool.add("two", prior_space, weight=0.6)

    assert sum(pool.counts(n).values()) == n


def test_the_split_follows_the_weights(search_space, prior_space):
    pool = SamplingPool(search_space, default_weight=0.5)
    pool.add("prior", prior_space, weight=0.5)

    assert pool.counts(100) == {"default": 50, "prior": 50}

    pool.set_weight("prior", 1.5)
    assert pool.counts(100) == {"default": 25, "prior": 75}


def test_a_source_weighted_out_gets_nothing(search_space, prior_space):
    pool = SamplingPool(search_space)
    pool.add("prior", prior_space, weight=0.0)

    assert pool.counts(10) == {"default": 10, "prior": 0}


def test_the_search_space_catches_a_pool_weighted_out_entirely(search_space, prior_space):
    """Sampling nothing at all would stall the maximizer, which is worse than ignoring the weights."""
    pool = SamplingPool(search_space, default_weight=0.0)
    pool.add("prior", prior_space, weight=0.0)

    assert pool.counts(10) == {"default": 10, "prior": 0}


def test_candidates_record_where_they_came_from(search_space, prior_space):
    pool = SamplingPool(search_space, default_weight=0.5)
    pool.add("prior", prior_space, weight=0.5)

    origins = {config.origin for config in pool.sample(10)}

    assert origins == {
        "Acquisition Function Maximizer: Random Search (default)",
        "Acquisition Function Maximizer: Random Search (prior)",
    }


def test_sources_can_be_added_and_removed_at_any_time(search_space, prior_space):
    pool = SamplingPool(search_space)

    pool.add("prior", prior_space)
    assert sorted(pool.sources) == ["default", "prior"]

    pool.remove("prior")
    assert sorted(pool.sources) == ["default"]

    with pytest.raises(KeyError, match="No sampling source"):
        pool.remove("prior")


def test_the_search_space_itself_cannot_be_displaced(search_space, prior_space):
    pool = SamplingPool(search_space)

    with pytest.raises(ValueError, match="reserved for the search space"):
        pool.add("default", prior_space)

    with pytest.raises(ValueError, match="cannot be removed"):
        pool.remove("default")


def test_negative_weights_are_rejected(search_space, prior_space):
    with pytest.raises(ValueError, match="must not be negative"):
        SamplingPool(search_space, default_weight=-1.0)

    pool = SamplingPool(search_space)

    with pytest.raises(ValueError, match="must not be negative"):
        pool.add("prior", prior_space, weight=-1.0)

    with pytest.raises(ValueError, match="must not be negative"):
        pool.set_weight("default", -1.0)


def test_a_plain_random_search_does_not_take_extra_spaces(search_space, prior_space):
    search = RandomSearch(configspace=search_space, acquisition_function=Acquisition())

    assert search.supports_sampling_spaces is False

    # Adding one creates the pool, so a prior supplied during a run is still sampled from.
    search.add_sampling_space("prior", prior_space)

    assert search.supports_sampling_spaces is True
    assert sorted(search.sampling_pool.sources) == ["default", "prior"]


def test_a_maximizer_that_cannot_sample_from_a_prior_says_so(search_space, prior_space):
    """Silently never sampling from a supplied prior would be worse than failing."""
    search = LocalSearch(configspace=search_space, acquisition_function=Acquisition())

    assert search.supports_sampling_spaces is False

    with pytest.raises(NotImplementedError, match="cannot sample from additional"):
        search.add_sampling_space("prior", prior_space)


def test_the_default_maximizer_can_be_given_a_prior_while_it_runs(search_space, prior_space):
    search = LocalAndSortedRandomSearch(
        configspace=search_space,
        acquisition_function=Acquisition(),
        max_steps=2,
        n_steps_plateau_walk=2,
        local_search_iterations=2,
    )

    assert search.supports_sampling_spaces is True

    search.add_sampling_space("prior", prior_space, weight=1.0)
    values = search._maximize(previous_configs=[], n_points=20)

    assert len(values) > 0

    search.remove_sampling_space("prior")
    assert sorted(search._random_search.sampling_pool.sources) == ["default"]


def test_the_legacy_uniform_split_still_works(search_space, prior_space):
    """`uniform_configspace` and `prior_sampling_fraction` are now two sources with a fixed split."""
    search = LocalAndSortedRandomSearch(
        configspace=prior_space,
        uniform_configspace=search_space,
        prior_sampling_fraction=0.25,
        acquisition_function=Acquisition(),
        max_steps=2,
        n_steps_plateau_walk=2,
        local_search_iterations=2,
    )

    assert search._random_search.sampling_pool.counts(100) == {"default": 25, "uniform": 75}
