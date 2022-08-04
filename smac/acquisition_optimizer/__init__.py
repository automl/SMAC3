from __future__ import annotations

import abc
from typing import Callable, Iterator, List, Optional, Tuple

import copy
import logging

import numpy as np

from acquisition_optimizer.differential_evolution import DifferentialEvolution
from acquisition_optimizer.local_and_random_search import LocalAndSortedRandomSearch
from acquisition_optimizer.local_search import LocalSearch
from acquisition_optimizer.random_search import RandomSearch
from smac.acquisition_function import AbstractAcquisitionFunction
from smac.chooser.random_chooser import ChooserNoCoolDown, RandomChooser
from smac.configspace import Configuration, ConfigurationSpace
from smac.runhistory.runhistory import RunHistory
from smac.utils.stats import Stats

__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"


__all__ = [
    "AbstractAcquisitionOptimizer",
    "DifferentialEvolution",
    "LocalAndSortedRandomSearch",
    "LocalSearch",
    "RandomSearch",
]


class AbstractAcquisitionOptimizer(metaclass=abc.ABCMeta):
    """Abstract class for the acquisition maximization.

    In order to use this class it has to be subclassed and the method
    ``_maximize`` must be implemented.

    Parameters
    ----------
    configspace : ConfigurationSpace
    acquisition_function : AbstractAcquisitionFunction
    challengers : int, defaults to 0
    seed : int, defaults to 0
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        seed: int = 0,
    ):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.acquisition_function = acquisition_function
        self.configspace = configspace
        self.challengers = challengers
        self.rng = np.random.RandomState(seed=seed)

    def _set_acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        self.acquisition_function = acquisition_function

    def maximize(
        self,
        previous_configs: List[Configuration],
        num_points: int | None = None,
        random_configuration_chooser: RandomChooser | None = None,
    ) -> Iterator[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        previous_configs: List[Configuration]
            Previous evaluated configurations.
        num_points: int
            Number of points to be sampled. If `num_points` is not specified, `self.challengers` is used.
        random_configuration_chooser: ~smac.optimizer.random_configuration_chooser.RandomConfigurationChooser, optional
            part of the returned ChallengerList such
            that we can interleave random configurations
            by a scheme defined by the random_configuration_chooser;
            random_configuration_chooser.next_smbo_iteration()
            is called at the end of this function

        Returns
        -------
        challengers : Iterator[Configuration]
            An iterable consisting of :class:`smac.configspace.Configuration`.
        """
        if num_points is None:
            num_points = self.challengers

        def next_configs_by_acq_value() -> List[Configuration]:
            assert num_points is not None
            return [t[1] for t in self._maximize(previous_configs, num_points)]

        challengers = ChallengerList(next_configs_by_acq_value, self.configspace, random_configuration_chooser)

        if random_configuration_chooser is not None:
            random_configuration_chooser.next_smbo_iteration()

        return challengers

    @abc.abstractmethod
    def _maximize(
        self,
        previous_configs: List[Configuration],
        num_points: int,
    ) -> List[Tuple[float, Configuration]]:
        """Implements acquisition function maximization.

        In contrast to ``maximize``, this method returns an iterable of tuples,
        consisting of the acquisition function value and the configuration. This
        allows to plug together different acquisition function maximizers.

        Parameters
        ----------
        previous_configs: List[Configuration]
            Previously evaluated configurations.
        num_points: int
            Number of points to be sampled.

        Returns
        -------
        challengers : List[Tuple[float, Configuration]]
            A list consisting of Tuple(acquisition_value, :class:`smac.configspace.Configuration`).
        """
        raise NotImplementedError()

    def _sort_configs_by_acq_value(self, configs: List[Configuration]) -> List[Tuple[float, Configuration]]:
        """Sort the given configurations by acquisition value.

        Parameters
        ----------
        configs : list(Configuration)

        Returns
        -------
        challengers : List[Tuple[float, Configuration]]
            Candidates ordered by their acquisition value.
        """
        assert self.acquisition_function is not None
        acq_values = self.acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]


class ChallengerList(Iterator):
    """Helper class to interleave random configurations in a list of challengers.

    Provides an iterator which returns a random configuration in each second
    iteration. Reduces time necessary to generate a list of new challengers
    as one does not need to sample several hundreds of random configurations
    in each iteration which are never looked at.

    Parameters
    ----------
    challenger_callback : Callable
        Callback function which returns a list of challengers (without interleaved random configurations, must a be a
        closure: https://www.programiz.com/python-programming/closure)

    configuration_space : ConfigurationSpace
        ConfigurationSpace from which to sample new random configurations.
    """

    def __init__(
        self,
        challenger_callback: Callable,
        configuration_space: ConfigurationSpace,
        random_configuration_chooser: Optional[RandomChooser] = ChooserNoCoolDown(modulus=2.0),
    ):
        self.challengers_callback = challenger_callback
        self.challengers = None  # type: Optional[List[Configuration]]
        self.configuration_space = configuration_space
        self._index = 0
        self._iteration = 1  # 1-based to prevent from starting with a random configuration
        self.random_configuration_chooser = random_configuration_chooser

    def __next__(self) -> Configuration:
        if self.challengers is not None and self._index == len(self.challengers):
            raise StopIteration
        elif self.random_configuration_chooser is None:
            if self.challengers is None:
                self.challengers = self.challengers_callback()
            config = self.challengers[self._index]
            self._index += 1
            return config
        else:
            if self.random_configuration_chooser.check(self._iteration):
                config = self.configuration_space.sample_configuration()
                config.origin = "Random Search"
            else:
                if self.challengers is None:
                    self.challengers = self.challengers_callback()
                config = self.challengers[self._index]
                self._index += 1
            self._iteration += 1
            return config

    def __len__(self) -> int:
        if self.challengers is None:
            self.challengers = self.challengers_callback()
        return len(self.challengers) - self._index


class FixedSet(AbstractAcquisitionOptimizer):
    def __init__(
        self,
        configurations: List[Configuration],
        acquisition_function: AbstractAcquisitionFunction,
        configspace: ConfigurationSpace,
        challengers: int = 5000,
        seed: int = 0,
    ):
        """Maximize the acquisition function over a finite list of configurations.

        Parameters
        ----------
        configurations : List[~smac.configspace.Configuration]
            Candidate configurations
        acquisition_function : ~smac.acquisition.AbstractAcquisitionFunction

        configspace : ~smac.configspace.ConfigurationSpace

        rng : np.random.RandomState or int, optional
        """
        super().__init__(
            acquisition_function=acquisition_function, configspace=configspace, challengers=challengers, seed=seed
        )
        self.configurations = configurations

    def _maximize(
        self,
        runhistory: RunHistory,
        stats: Stats,
        num_points: int,
    ) -> List[Tuple[float, Configuration]]:
        configurations = copy.deepcopy(self.configurations)
        for config in configurations:
            config.origin = "Fixed Set"
        return self._sort_configs_by_acq_value(configurations)
