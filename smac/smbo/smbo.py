import logging
import numpy as np

from ParameterConfigSpace.config_space import ConfigSpace

from esmac.local_search import LocalSearch

from robo.acquisition.ei import EI
from robo.models.random_forest import RandomForestWithInstances
from robo.recommendation.incumbent import compute_incumbent

from robo.solver.base_solver import BaseSolver


class SMBO(BaseSolver):

    def __init__(self, pcs_file, instance_features, seed=42):

        self.config_space = ConfigSpace(pcs_file)
        self.instance_features = instance_features

        # Extract types vector for rf from config space
        self.types = np.zeros(len(self.config_space.get_parameter_names()))

        # Extract bounds of the input space
        X_lower = np.zeros([self.types.shape[0]])
        X_upper = np.zeros([self.types.shape[0]])

        for i, param in enumerate(self.config_space.get_parameter_names()):
            if param in self.config_space.get_categorical_parameters():
                n_cats = len(self.config_space.get_categorical_values(param))
                self.types[i] = n_cats
                X_lower[i] = 0
                X_upper[i] = n_cats
            elif param in self.config_space.get_continuous_parameters():
                lo, up = self.config_space.parameters[param].values
                X_lower[i] = lo
                X_upper[i] = up

        self.model = RandomForestWithInstances(self.types,
                                               self.instance_features)

        self.acquisition_func = EI(self.model,
                                   X_lower,
                                   X_upper,
                                   compute_incumbent)

        self.local_search = LocalSearch(self.acquisition_func,
                                        self.config_space)
        self.seed = seed

    def run(self, max_iters=10):
        '''
            Runs the Bayesian optimization loop for max_iters iterations
            Args:
                max_iters : The maximum number of iterations (int)
            Return:
                incumbent: the global optimizer (2D numpy array)
        '''

        #Initialize X, Y
        for i in range(max_iters):
            next_config = self.choose_next()
            #incumbent = itensify()
            #Evaluate nex_config and update X, Y

        incumbent = np.random.randn(1, 1)
        return incumbent

    def choose_next(self, X=None, Y=None, n_iters=10):
        '''
            Chooses the next configuration by training the model and
            optimizing the acquisition function.
            Args:
                X : The configuration we have seen so far (2D numpy array)
                Y : The function values of the configurations (2D numpy array)
            Return:
                incumbent: The next configuration to evaluate (2D numpy array)
        '''
        self.model.train(X, Y)
        self.acquisition_func.update(self.model)

        found_configs = []
        acq_vals = np.zeros([n_iters])

        # Start N local search from different random start points
        for i in range(n_iters):
            start_point = self.config_space.get_random_config_vector()
            configuration, acq_val = self.local_search.maximize(start_point)

            found_configs.append(configuration)
            acq_vals[i] = acq_val[0][0]

        # Return configuration with highest acquisition value
        best = np.argmax(acq_vals)
        return found_configs[best]
