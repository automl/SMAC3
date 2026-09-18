from __future__ import annotations

from typing import Any, Mapping, Protocol

from dataclasses import dataclass

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)

DEFAULT_SOURCE = "default"


class Sampler(Protocol):
    """Anything candidates can be drawn from.

    Both a `ConfigurationSpace` and a user prior can produce configurations, and the pool does not care which it
    is holding. A prior with no tractable sampler returns `None` and simply contributes no candidates.
    """

    def sample(self, n: int, rng: np.random.RandomState | None = None) -> list[Configuration] | None:
        """Draws `n` configurations."""
        ...


class ConfigSpaceSampler:
    """Draws candidates from a configuration space."""

    def __init__(self, configspace: ConfigurationSpace) -> None:
        self.configspace = configspace

    def sample(self, n: int, rng: np.random.RandomState | None = None) -> list[Configuration] | None:
        """Draws `n` configurations from the configuration space."""
        if n == 1:
            return [self.configspace.sample_configuration()]

        return list(self.configspace.sample_configuration(size=n))


def as_sampler(source: Sampler | ConfigurationSpace) -> Sampler:
    """Returns something candidates can be drawn from, wrapping a configuration space if that is what was given."""
    if isinstance(source, ConfigurationSpace):
        return ConfigSpaceSampler(source)

    if not hasattr(source, "sample"):
        raise TypeError(f"{source!r} is neither a configuration space nor something that can be sampled from.")

    return source


@dataclass(frozen=True)
class SamplingSource:
    """One place to draw candidates from, and how much of the budget it gets."""

    sampler: Sampler
    weight: float = 1.0


class SamplingPool:
    """A weighted mixture of configuration spaces to draw random candidates from.

    An acquisition function maximizer samples candidates and ranks them. That works while the interesting region
    is large enough to stumble into, and stops working when it is not: a sharply peaked user prior can raise the
    acquisition value enormously in a region that uniform sampling will never visit. Drawing part of the candidates
    from the prior itself is what makes such a belief usable at all.

    Sources are keyed and can be added or removed while the optimization is running, because a user may supply a
    belief at any point during a run.

    Parameters
    ----------
    default : ConfigurationSpace
        The search space itself, which is always one of the sources.
    default_weight : float, defaults to 1.0
        Share of the budget reserved for the search space, relative to the other sources.
    seed : int, defaults to 0
        Random seed, used to break ties when splitting the budget.
    """

    def __init__(
        self,
        default: Sampler | ConfigurationSpace,
        *,
        default_weight: float = 1.0,
        seed: int = 0,
    ) -> None:
        if default_weight < 0:
            raise ValueError(f"A sampling weight must not be negative, got {default_weight}.")

        self._sources: dict[str, SamplingSource] = {
            DEFAULT_SOURCE: SamplingSource(sampler=as_sampler(default), weight=default_weight)
        }
        self._seed = seed
        self._rng = np.random.RandomState(seed=seed)

    @property
    def sources(self) -> Mapping[str, SamplingSource]:
        """The registered sources, by key."""
        return dict(self._sources)

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "seed": self._seed,
            "weights": {key: source.weight for key, source in self._sources.items()},
        }

    def add(self, key: str, source: Sampler | ConfigurationSpace, weight: float = 1.0) -> None:
        """Registers a configuration space, or a prior, to draw part of the candidates from."""
        if key == DEFAULT_SOURCE:
            raise ValueError(f"The key {DEFAULT_SOURCE!r} is reserved for the search space itself.")

        if weight < 0:
            raise ValueError(f"A sampling weight must not be negative, got {weight}.")

        self._sources[key] = SamplingSource(sampler=as_sampler(source), weight=weight)

    def remove(self, key: str) -> None:
        """Removes a previously registered source."""
        if key == DEFAULT_SOURCE:
            raise ValueError("The search space itself cannot be removed from the pool.")

        if key not in self._sources:
            raise KeyError(f"No sampling source is registered under the key {key!r}.")

        del self._sources[key]

    def set_weight(self, key: str, weight: float) -> None:
        """Changes the share of the budget a source gets."""
        if key not in self._sources:
            raise KeyError(f"No sampling source is registered under the key {key!r}.")

        if weight < 0:
            raise ValueError(f"A sampling weight must not be negative, got {weight}.")

        self._sources[key] = SamplingSource(sampler=self._sources[key].sampler, weight=weight)

    def counts(self, n: int) -> dict[str, int]:
        """Splits `n` candidates across the sources, proportionally to their weights.

        Uses largest remainder rounding, so the counts always add up to exactly `n` however many sources there
        are and however the weights divide.
        """
        if n <= 0:
            return {key: 0 for key in self._sources}

        weights = np.array([source.weight for source in self._sources.values()], dtype=float)
        total = weights.sum()

        if total <= 0:
            # Every source has been weighted out; fall back to the search space rather than sampling nothing.
            return {key: (n if key == DEFAULT_SOURCE else 0) for key in self._sources}

        exact = n * weights / total
        counts = np.floor(exact).astype(int)
        remaining = n - int(counts.sum())

        if remaining > 0:
            # Largest fractional part first; a stable order keeps the split reproducible.
            order = np.argsort(-(exact - counts), kind="stable")
            counts[order[:remaining]] += 1

        return {key: int(count) for key, count in zip(self._sources, counts)}

    def sample(self, n: int) -> list[Configuration]:
        """Draws `n` configurations, split across the sources.

        Each configuration is tagged with the source it came from, so that a candidate which made it through can
        be traced back to the belief that suggested it.
        """
        configurations: list[Configuration] = []

        for key, count in self.counts(n).items():
            if count <= 0:
                continue

            drawn = self._sources[key].sampler.sample(count, self._rng)

            if drawn is None:
                # A prior with no tractable sampler. It still weights the acquisition function.
                logger.debug(f"The sampling source {key!r} cannot be sampled from; it contributes no candidates.")
                continue

            for configuration in drawn:
                configuration.origin = f"Acquisition Function Maximizer: Random Search ({key})"

            configurations.extend(drawn)

        return configurations
