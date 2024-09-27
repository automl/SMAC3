from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterator

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer.helpers import ChallengerList
from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractAcquisitionMaximizer:
    """Abstract class for the acquisition maximization.

    In order to use this class it has to be subclassed and the
    method `_maximize` must be implemented.

    Parameters
    ----------
    configspace : ConfigurationSpace acquisition_function : AbstractAcquisitionFunction
    challengers : int, defaults to 5000 Number of configurations sampled during the optimization process,
    details depend on the used maximizer. Also, the number of configurations that is returned by calling `maximize`.
    seed : int, defaults to 0
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        seed: int = 0,
    ):
        self._configspace = configspace
        self._acquisition_function = acquisition_function
        self._challengers = challengers
        self._seed = seed
        self._rng = np.random.RandomState(seed=seed)

    @property
    def acquisition_function(self) -> AbstractAcquisitionFunction | None:
        """The acquisition function used for maximization."""
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        self._acquisition_function = acquisition_function

    @property
    def meta(self) -> dict[str, Any]:
        """Return the meta-data of the created object."""
        acquisition_function_meta = None
        if self._acquisition_function is not None:
            acquisition_function_meta = self._acquisition_function.meta

        return {
            "name": self.__class__.__name__,
            "acquisition_function": acquisition_function_meta,
            "challengers": self._challengers,
            "seed": self._seed,
        }

    def maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int | None = None,
        random_design: AbstractRandomDesign | None = None,
    ) -> Iterator[Configuration]:
        """Maximize acquisition function using `_maximize`, implemented by a subclass.

        Parameters
        ----------
        previous_configs: list[Configuration]
            Previous evaluated configurations.
        n_points: int, defaults to None
            Number of points to be sampled & number of configurations to be returned. If `n_points` is not specified,
            `self._challengers` is used. Semantics depend on concrete implementation.
        random_design: AbstractRandomDesign, defaults to None
            Part of the returned ChallengerList such that we can interleave random configurations
            by a scheme defined by the random design. The method `random_design.next_iteration()`
            is called at the end of this function.

        Returns
        -------
        challengers : Iterator[Configuration]
            An iterable consisting of configurations.
        """
        if n_points is None:
            n_points = self._challengers

        def next_configs_by_acquisition_value() -> list[Configuration]:
            assert n_points is not None
            # since maximize returns a tuple of acquisition value and configuration,
            # and we only need the configuration, we return the second element of the tuple
            # for each element in the list
            return [t[1] for t in self._maximize(previous_configs, n_points)]

        challengers = ChallengerList(
            self._configspace,
            next_configs_by_acquisition_value,
            random_design,
        )

        if random_design is not None:
            random_design.next_iteration()

        return challengers

    @abstractmethod
    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        """Implement acquisition function maximization.

        In contrast to `maximize`, this method returns an iterable of tuples, consisting of the acquisition function
        value and the configuration. This allows to plug together different acquisition function maximizers.

        Parameters
        ----------
        previous_configs: list[Configuration]
            Previously evaluated configurations.
        n_points: int
            Number of points to be sampled.

        Returns
        -------
        challengers : list[tuple[float, Configuration]]
            A list consisting of tuples of acquisition_value and its configuration.
        """
        raise NotImplementedError()

    def _sort_by_acquisition_value(self, configs: list[Configuration]) -> list[tuple[float, Configuration]]:
        """Sort the given configurations by the acquisition value.

        Parameters
        ----------
        configs : list[Configuration]

        Returns
        -------
        challengers : list[tuple[float, Configuration]]
            Candidates ordered by their acquisition value (descending).
        """
        assert self._acquisition_function is not None
        acq_values = self._acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self._rng.rand(len(acq_values))

        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]
