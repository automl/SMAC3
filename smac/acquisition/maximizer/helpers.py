from __future__ import annotations

from typing import Callable, Iterator

from ConfigSpace import Configuration, ConfigurationSpace

from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.random_design.modulus_design import ModulusRandomDesign


class ChallengerList(Iterator):
    """Helper class to interleave random configurations in a list of challengers.

    Provides an iterator which returns a random configuration in each second
    iteration. Reduces time necessary to generate a list of new challengers
    as one does not need to sample several hundreds of random configurations
    in each iteration which are never looked at.

    Parameters
    ----------
    configspace : ConfigurationSpace
    challenger_callback : Callable
        Callback function which returns a list of challengers (without interleaved random configurations, must a be a
        closure: https://www.programiz.com/python-programming/closure)
    random_design : AbstractRandomDesign | None, defaults to ModulusRandomDesign(modulus=2.0)
        Which random design should be used.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        challenger_callback: Callable,
        random_design: AbstractRandomDesign | None = ModulusRandomDesign(modulus=2.0),
    ):
        self._challengers_callback = challenger_callback
        self._challengers: list[Configuration] | None = None
        self._configspace = configspace
        self._index = 0
        self._iteration = 1  # 1-based to prevent from starting with a random configuration
        self._random_design = random_design

    def __next__(self) -> Configuration:
        if self._challengers is not None and self._index == len(self._challengers):
            raise StopIteration
        elif self._random_design is None:
            if self._challengers is None:
                self._challengers = self._challengers_callback()

            config = self._challengers[self._index]
            self._index += 1

            return config
        else:
            if self._random_design.check(self._iteration):
                config = self._configspace.sample_configuration()
                config.origin = "Random Search"
            else:
                if self._challengers is None:
                    self._challengers = self._challengers_callback()

                config = self._challengers[self._index]
                self._index += 1
            self._iteration += 1

            return config

    def __len__(self) -> int:
        if self._challengers is None:
            self._challengers = self._challengers_callback()

        return len(self._challengers) - self._index


'''
class FixedSet(AbstractAcquisitionMaximizer):
    def __init__(
        self,
        configurations: list[Configuration],
        acquisition_function: AbstractAcquisitionFunction,
        configspace: ConfigurationSpace,
        challengers: int = 5000,
        seed: int = 0,
    ):
        """Maximize the acquisition function over a finite list of configurations.

        Parameters
        ----------
        configurations : list[~smac._configspace.Configuration]
            Candidate configurations
        acquisition_function : ~smac.acquisition.AbstractAcquisitionFunction

        configspace : ~smac._configspace.ConfigurationSpace

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
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        configurations = copy.deepcopy(self.configurations)
        for config in configurations:
            config.origin = "Fixed Set"

        return self._sort_by_acquisition_value(configurations)
'''
