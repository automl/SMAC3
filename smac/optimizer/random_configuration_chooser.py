from abc import ABC, abstractmethod
import logging


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

    @abstractmethod
    def next_smbo_iteration(self):
        """Indicate beginning of next SMBO iteration"""
        pass

    @abstractmethod
    def check(self, iteration):
        """Check if iteration should be followed by a random configuration"""
        pass


class ChooserNoCoolDown(RandomConfigurationChooser):

    def __init__(self, modulus: float=2.0):
        if modulus <= 1.0:
            logging.warning("Using SMAC with random configurations only."
                            "ROAR is the better choice for this.")
        self.modulus = modulus

    def next_smbo_iteration(self):
        pass

    def check(self, iteration):
        return iteration % self.modulus < 1


class ChooserLinearCoolDown(RandomConfigurationChooser):

    def __init__(self, start_modulus: float=2.0, modulus_increment: float=0.3, end_modulus: float=float('inf')):
        if start_modulus <= 1.0 and modulus_increment <= 0.0:
            logging.warning("Using SMAC with random configurations only."
                            "ROAR is the better choice for this.")
        self.modulus = start_modulus
        self.modulus_increment = modulus_increment
        self.end_modulus = end_modulus
        self.last_iteration = 0

    def next_smbo_iteration(self):
        self.modulus += self.modulus_increment
        self.modulus = min(self.modulus, self.end_modulus)
        self.last_iteration = 0

    def check(self, iteration: int):
        if (iteration - self.last_iteration) % self.modulus < 1:
            self.last_iteration = iteration
            return True
        else:
            return False
