import logging
import time
import random
import numpy as np

from smac.configspace import impute_inactive_values, get_one_exchange_neighbourhood, Configuration

__author__ = "Aaron Klein, Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


class LocalSearch(object):

    def __init__(self, acquisition_function, config_space,
                 epsilon=0.000001, max_iterations=None, rng=None):
        """
        Implementation of SMAC's local search

        Parameters:
        ----------

        acquisition_function:  function
            The function which the local search tries to maximize
        config_space:  ConfigSpace
            Parameter configuration space
        epsilon: float
            In order to perform a local move one of the incumbent's neighbors
            needs at least an improvement higher than epsilon
        max_iterations: int
            Maximum number of iterations that the local search will perform
        """
        self.config_space = config_space
        self.acquisition_function = acquisition_function
        self.epsilon = epsilon
        self.max_iterations = max_iterations

        if rng is None:
            self.rng = np.random.RandomState(seed=np.random.randint(10000))
        else:
            self.rng = rng

        self.logger = logging.getLogger("localsearch")

    def maximize(self, start_point, *args):
        """
        Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters:
        ----------

        start_point:  np.array(1, D):
            The point from where the local search starts
        *args :
            Additional parameters that will be passed to the
            acquisition function

        Returns:
        -------

        incumbent np.array(1, D):
            The best found configuration
        acq_val_incumbent np.array(1,1) :
            The acquisition value of the incumbent

        """
        incumbent = start_point
        # Compute the acquisition value of the incumbent
        incumbent_ = impute_inactive_values(incumbent)
        acq_val_incumbent = self.acquisition_function(
                                            incumbent_.get_array(),
                                            *args)

        local_search_steps = 0
        neighbors_looked_at = 0
        time_n = []
        while True:

            local_search_steps += 1
            if local_search_steps % 1000 == 0:
                self.logger.warn("Local search took already %d iterations." \
                "Is it maybe stuck in a infinite loop?", local_search_steps)

            # Get neighborhood of the current incumbent
            # by randomly drawing configurations
            changed_inc = False

            all_neighbors = get_one_exchange_neighbourhood(incumbent,
                                                    seed=self.rng.seed())
            self.rng.shuffle(all_neighbors)

            for neighbor in all_neighbors:
                s_time = time.time()
                neighbor_ = impute_inactive_values(neighbor)
                n_array = neighbor_.get_array()

                acq_val = self.acquisition_function(n_array, *args)

                neighbors_looked_at += 1

                time_n.append(time.time() - s_time)

                if acq_val > acq_val_incumbent + self.epsilon:
                    self.logger.debug("Switch to one of the neighbors")
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    break

            if (not changed_inc) or (self.max_iterations != None
                                     and local_search_steps == self. max_iterations):
                self.logger.debug("Local search took %d steps and looked at %d configurations. "
                                  "Computing the acquisition value for one "
                                  "configuration took %f seconds on average.",
                                  local_search_steps, neighbors_looked_at, np.mean(time_n))
                break

        return incumbent, acq_val_incumbent
