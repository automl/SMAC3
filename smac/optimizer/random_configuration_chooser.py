from abc import ABC, abstractmethod
import logging

import numpy as np

__author__ = "Aaron Kimmig"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Kimmig"
__email__ = "kimmiga@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RandomConfigurationChooser(ABC):
    """
    Abstract base of helper classes to configure interleaving of
    random configurations in a list of challengers.
    """

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng

    @abstractmethod
    def next_smbo_iteration(self) -> None:
        """Indicate beginning of next SMBO iteration"""
        pass

    @abstractmethod
    def check(self, iteration) -> bool:
        """Check if the next configuration should be at random"""
        pass


class ChooserNoCoolDown(RandomConfigurationChooser):
    """Interleave a random configuration after a constant number of configurations found by Bayesian optimization.

    Parameters
    ----------
    modulus : float
        Every modulus-th configuration will be at random.

    """

    def __init__(self, rng: np.random.RandomState, modulus: float = 2.0):
        super().__init__(rng)

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        if modulus <= 1.0:
            self.logger.warning("Using SMAC with random configurations only."
                                "ROAR is the better choice for this.")
        self.modulus = modulus

    def next_smbo_iteration(self) -> None:
        pass

    def check(self, iteration) -> bool:
        return iteration % self.modulus < 1


class ChooserProb(RandomConfigurationChooser):

    def __init__(self, rng: np.random.RandomState, prob: float):
        """Interleave a random configuration according to a given probability.

        Parameters
        ----------
        prob : float
            Probility of a random configuration
        rng : np.random.RandomState
            Random state
        """
        super().__init__(rng)
        self.prob = prob

    def next_smbo_iteration(self) -> None:
        pass

    def check(self, iteration: int) -> bool:
        if self.rng.rand() < self.prob:
            return True
        else:
            return False
