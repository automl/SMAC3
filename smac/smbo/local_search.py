'''
Created on Mar 27, 2015

@author: Aaron Klein
'''
import logging
import time
import numpy as np

from robo.maximizers.base_maximizer import BaseMaximizer


class LocalSearch(BaseMaximizer):

    def __init__(self, acquisition_function, config_space, epsilon=0.01, n_neighbours=20, max_iterations=None):
        '''
        Constructor
        '''
        self.config_space = config_space
        self.acquisition_function = acquisition_function
        self.epsilon = epsilon
        self.n_neighbours = n_neighbours
        self.max_iterations = max_iterations

    def maximize(self, start_point, *args):
        """
            Implementation of SMAC's local search

            Parameters:
            ----------

            start_point (vector) : The point where the local search starts from
            acquisition_function (function) : The function the local search tries to maximize
            config_space (ConfigSpace) : Parameter configuration space
            epsilon (float) : In order to perform a local move one of the incumbent's neighbours needs an improvement that is higher than epsilon
            n_neighbours (int) : Number of neighbours that will be samples in each local move
            max_iterations (int) : Maximum number of iterations that the local search will perform
            *args : Additional parameters that will be passed to the acquisition function

            Returns:
            -------

            incumbent (vector) : The best found configuration
            acq_val_incumbent (float) : The acquisition value of the incumbent

        """
        incumbent = start_point
        local_search_steps = 0

        while True:

            local_search_steps += 1
            if local_search_steps % 1000 == 0:
                logging.warn("Local search took already %d iterations. \
                Is it maybe stuck in a infinite loop?", local_search_steps)

            # Compute the acquisition value of the incumbent
            incumbent = self.config_space.impute_non_active(incumbent, value="def")
            acq_val_incumbent = self.acquisition_function(incumbent, *args)

            # Get neighbourhood of the current incumbent by randomly drawing configurations
            neighbourhood = np.zeros([self.n_neighbours, incumbent.shape[0]])
            for i in range(self.n_neighbours):
                n = self.config_space.get_random_neighbor(incumbent)
                neighbourhood[i] = self.config_space.impute_non_active(n, value="def")

            # Compute acquisition values for all points in the neighbourhood
            acq_val_neighbours = np.zeros([self.n_neighbours])
            t_acq = 0
            n_acq = 0
            for i in range(self.n_neighbours):
                #neighbour_config = self.config_space.impute_non_active(neighbourhood[i], value="def")
                s = time.time()
                acq_val_neighbours[i] = self.acquisition_function(neighbourhood[i], *args)
                t_acq += time.time() - s
                n_acq += 1

            # Determine the best neighbour with highest acquisition value
            acq_val_best = np.max(acq_val_neighbours)

            # If no significant improvement break, otherwise move to one of the best neighbours
            if acq_val_best > acq_val_incumbent + self.epsilon:

                logging.info("Switch to one of the neighbours")
                # List of best neighbours
                best_indices = np.where(acq_val_neighbours > acq_val_incumbent + self.epsilon)[0]

                best_neighbours = np.zeros([best_indices.shape[0], neighbourhood.shape[1]])
                for i in range(best_indices.shape[0]):
                    best_neighbours[i] = neighbourhood[best_indices[i]]

                # Move to one of the best neighbours randomly
                random_idx = np.random.randint(0, best_indices.shape[0])
                incumbent = best_neighbours[random_idx]

            else:
                logging.info("Local search took %d steps. Computing the acquisition value for one configuration took %f seconds on average.", local_search_steps, (t_acq / float(n_acq)))
                break

            if self.max_iterations != None and local_search_steps == self. max_iterations:
                logging.info("Local search took %d steps. Computing the acquisition value for one configuration took %f seconds on average.", local_search_steps, (t_acq / float(n_acq)))
                break

        return incumbent, acq_val_incumbent
