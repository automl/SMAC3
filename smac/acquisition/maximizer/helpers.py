from __future__ import annotations

from typing import Callable, Iterator

from ConfigSpace import Configuration, ConfigurationSpace

from smac.random_design import ProbabilityRandomDesign
from smac.random_design.abstract_random_design import AbstractRandomDesign


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
        Callback function which returns a list of challengers (without interleaved random configurations), must a be a
        python closure.
    random_design : AbstractRandomDesign | None, defaults to ModulusRandomDesign(modulus=2.0)
        Which random design should be used.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        challenger_callback: Callable,
        random_design: AbstractRandomDesign | None = ProbabilityRandomDesign(seed=0, probability=0.08447232371720552),
    ):
        self._challengers_callback = challenger_callback
        self._challengers: list[Configuration] | None = None
        self._configspace = configspace
        self._index = 0
        self._iteration = 1  # 1-based to prevent from starting with a random configuration
        self._random_design = random_design

    def __next__(self) -> Configuration:
        # If we already returned the required number of challengers
        if self._challengers is not None and self._index == len(self._challengers):
            raise StopIteration
        # If we do not want to have random configs, we just yield the next challenger
        elif self._random_design is None:
            if self._challengers is None:
                self._challengers = self._challengers_callback()

            config = self._challengers[self._index]
            self._index += 1

            return config
        # If we want to interleave challengers with random configs, sample one
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
