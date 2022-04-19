from abc import ABC, abstractmethod
from typing import Optional

import logging

import numpy as np

__author__ = "Aaron Kimmig"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Kimmig"
__email__ = "kimmiga@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RandomConfigurationChooser(ABC):
    """Abstract base of helper classes to configure interleaving of random configurations in a list
    of challengers.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState(seed=0)

    @abstractmethod
    def next_smbo_iteration(self) -> None:
        """Indicate beginning of next SMBO iteration."""
        pass

    @abstractmethod
    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be at random."""
        pass


class ChooserNoCoolDown(RandomConfigurationChooser):
    """Interleave a random configuration after a constant number of configurations found by Bayesian
    optimization.

    Parameters
    ----------
    modulus : float
        Every modulus-th configuration will be at random.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None, modulus: float = 2.0):
        super().__init__(rng)

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        if modulus <= 1.0:
            self.logger.warning("Using SMAC with random configurations only." "ROAR is the better choice for this.")
        self.modulus = modulus

    def next_smbo_iteration(self) -> None:
        """Does nothing."""
        ...

    def check(self, iteration: int) -> bool:
        """Checks if the next configuration should be at random."""
        return iteration % self.modulus < 1


class ChooserLinearCoolDown(RandomConfigurationChooser):
    """Interleave a random configuration, decreasing the fraction of random configurations over
    time.

    Parameters
    ----------
    start_modulus : float
       Initially, every modulus-th configuration will be at random
    modulus_increment : float
       Increase modulus by this amount in every iteration
    end_modulus : float
       Highest modulus used in the chooser. If the value is reached before the optimization is over, it is not
       further increased. If it is not reached before the optimization is over, there will be no adjustment to make
       sure that the ``end_modulus`` is reached.
    """

    def __init__(
        self,
        rng: Optional[np.random.RandomState] = None,
        start_modulus: float = 2.0,
        modulus_increment: float = 0.3,
        end_modulus: float = np.inf,
    ):
        super().__init__(rng)

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        if start_modulus <= 1.0 and modulus_increment <= 0.0:
            self.logger.warning("Using SMAC with random configurations only. ROAR is the better choice for this.")
        self.modulus = start_modulus
        self.modulus_increment = modulus_increment
        self.end_modulus = end_modulus
        self.last_iteration = 0

    def next_smbo_iteration(self) -> None:
        """Change modulus."""
        self.modulus += self.modulus_increment
        self.modulus = min(self.modulus, self.end_modulus)
        self.last_iteration = 0

    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be interleaved based on modulus."""
        if (iteration - self.last_iteration) % self.modulus < 1:
            self.last_iteration = iteration
            return True
        else:
            return False


class ChooserProb(RandomConfigurationChooser):
    """Interleave a random configuration according to a given probability.

    Parameters
    ----------
    prob : float
        Probility of a random configuration
    rng : np.random.RandomState
        Random state
    """

    def __init__(self, rng: Optional[np.random.RandomState], prob: float):
        super().__init__(rng)
        self.prob = prob

    def next_smbo_iteration(self) -> None:
        """Does nothing."""
        ...

    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be at random."""
        if self.rng.rand() < self.prob:
            return True
        else:
            return False


class ChooserProbCoolDown(RandomConfigurationChooser):
    """Interleave a random configuration according to a given probability which is decreased over
    time.

    Parameters
    ----------
    prob : float
        Probility of a random configuration
    cool_down_fac : float
        Multiply the ``prob`` by ``cool_down_fac`` in each iteration
    rng : np.random.RandomState
        Random state
    """

    def __init__(self, rng: Optional[np.random.RandomState], prob: float, cool_down_fac: float):
        super().__init__(rng)
        self.prob = prob
        self.cool_down_fac = cool_down_fac

    def next_smbo_iteration(self) -> None:
        """Set the probability to the current value multiplied by the `cool_down_fac`."""
        self.prob *= self.cool_down_fac

    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be at random."""
        if self.rng.rand() < self.prob:
            return True
        else:
            return False


class ChooserCosineAnnealing(RandomConfigurationChooser):
    """Interleave a random configuration according to a given probability which is decreased
    according to a cosine annealing schedule.

    Parameters
    ----------
    prob_max : float
        Initial probility of a random configuration
    prob_min : float
        Lowest probility of a random configuration
    restart_iteration : int
        Restart the annealing schedule every ``restart_iteration`` iterations.
    rng : np.random.RandomState
        Random state
    """

    def __init__(
        self,
        rng: Optional[np.random.RandomState],
        prob_max: float,
        prob_min: float,
        restart_iteration: int,
    ):
        super().__init__(rng)
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.prob_max = prob_max
        self.prob_min = prob_min
        self.restart_iteration = restart_iteration
        self.iteration = 0
        self.prob = prob_max

    def next_smbo_iteration(self) -> None:
        """Set `self.prob` and increases the iteration counter."""
        self.prob = self.prob_min + (
            0.5 * (self.prob_max - self.prob_min) * (1 + np.cos(self.iteration * np.pi / self.restart_iteration))
        )
        self.logger.error("Probability for random configs: %f" % self.prob)
        self.iteration += 1
        if self.iteration > self.restart_iteration:
            self.iteration = 0
            self.logger.error("Perform restart in next iteration!")

    def check(self, iteration: int) -> bool:
        """Check if a random configuration should be interleaved."""
        if self.rng.rand() < self.prob:
            self.logger.error("Random Config")
            return True
        else:
            self.logger.error("Acq Config")
            return False
