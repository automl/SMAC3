from __future__ import annotations

from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer.abstract_acquisition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.acquisition.maximizer.sampling import SamplingPool
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class RandomSearch(AbstractAcquisitionMaximizer):
    """Get candidate solutions via random sampling of configurations.

    Parameters
    ----------
    configspace : ConfigurationSpace
        Configuration space used for sampling.
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
        Acquisition function to maximize.
    challengers : int, defaults to 5000
        Number of configurations sampled during the optimization process.
    seed : int, defaults to 0
        Random seed.
    sampling_pool : SamplingPool | None, defaults to None
        Draws the candidates from a weighted mixture of configuration spaces rather than from the search space
        alone. Used to sample from user priors over the optimum, which can be supplied while the run is going on.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        seed: int = 0,
        sampling_pool: SamplingPool | None = None,
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            seed=seed,
        )

        self._sampling_pool = sampling_pool

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta

        if self._sampling_pool is not None:
            meta.update({"sampling_pool": self._sampling_pool.meta})

        return meta

    @property
    def sampling_pool(self) -> SamplingPool | None:
        """The mixture of configuration spaces candidates are drawn from, if there is one."""
        return self._sampling_pool

    @property
    def supports_sampling_spaces(self) -> bool:  # noqa: D102
        return self._sampling_pool is not None

    def add_sampling_space(  # noqa: D102
        self, key: str, configspace: ConfigurationSpace, weight: float | None = None
    ) -> None:
        if self._sampling_pool is None:
            self._sampling_pool = SamplingPool(self._configspace, seed=self._seed)

        self._sampling_pool.add(key, configspace, weight if weight is not None else 1.0)

    def remove_sampling_space(self, key: str) -> None:  # noqa: D102
        if self._sampling_pool is None:
            raise KeyError(f"No sampling source is registered under the key {key!r}.")

        self._sampling_pool.remove(key)

    def _sample_configurations(self, n_points: int) -> list[Configuration]:
        if self._sampling_pool is None:
            return super()._sample_configurations(n_points)

        return self._sampling_pool.sample(n_points)

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
        _sorted: bool = False,
    ) -> list[tuple[float, Configuration]]:
        """Maximize acquisition function with random search

        Parameters
        ----------
        previous_configs : list[Configuration]
            Not used.
        n_points : int
            Number of configurations to return.
        _sorted : bool, optional
            If True, sort candidates by their acquisition value (descending), by default False

        Returns
        -------
        list[tuple[float, Configuration]]
            Candidates with their acquisition function value. (acq value, candidate)
        """
        rand_configs = self._sample_configurations(n_points)

        if _sorted:
            origin = "Acquisition Function Maximizer: Random Search (sorted)"
        else:
            origin = "Acquisition Function Maximizer: Random Search"

        for config in rand_configs:
            # A configuration drawn from the pool already records which source suggested it.
            if getattr(config, "origin", None) is None:
                config.origin = origin

        if _sorted:
            return self._sort_by_acquisition_value(rand_configs)

        return [(0, config) for config in rand_configs]
