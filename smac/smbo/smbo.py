import logging
import numpy as np


from ParameterConfigSpace.config_space import ConfigSpace

from smac.smbo.local_search import LocalSearch
# Aaron: I already implemented the local search in ESMAC and EI in RoBO maybe
# we can reuse it here
from smac.smbo.ei import EI
from robo.models.random_forest import RandomForest
from robo.recommendation.incumbent import compute_incumbent
 
from robo.solver.base_solver import BaseSolver

class SMBO(BaseSolver):
    # Aaron: I would like to derive this class from the RoBO interface in order
    # to share the output and logging functionality between RoBO and SMAC.
    # Aaron: I will change the default value to RandomForest, EI and
    # LocalSearch as soon as we have configured the dependencies

#     def __init__(self, config_space,
#                  acquisition_func=None,
#                  maximizer=None,
#                  seed=42):
#         '''
#             Implementation of the main Bayesian optimization loop
#             Args:
#                 model : A model that captures our believe of our objective
#                     function (robo model object)
#                 acquisition_func : Surrogate function to pick a new
#                     configuration (robo acquisition object)
#                 maximizer: Optimization strategy to maximize the
#                     acquisition function (robo maximizer object)
#                 seed : random seed (integer)
#         '''
#         self.config_space = config_space
#         types = self.config_space.get_categorical_values()
#         self.model = RandomForest(types)
#         self.acquisition_func = EI(self.model, X_lower, X_upper, compute_incumbent)
#         self.maximizer = LocalSearch(self.acquisition_function, self.config_space)
#         self.seed = seed

    def __init__(self, pcs_file, seed=42):    
        '''
            Implementation of the main Bayesian optimization loop
            Args:
                model : A model that captures our believe of our objective
                    function (robo model object)
                acquisition_func : Surrogate function to pick a new
                    configuration (robo acquisition object)
                maximizer: Optimization strategy to maximize the
                    acquisition function (robo maximizer object)
                seed : random seed (integer)
        '''
        self.config_space = ConfigSpace(pcs_file)
        
        
        # Extract types vector for rf from config space
        self.types = np.zeros(len(self.config_space.get_parameter_names()))

        for i, param in enumerate(self.config_space.get_parameter_names()):
            if param in self.config_space.get_categorical_parameters():
                self.types[i] = len(self.config_space.get_categorical_values(param))
        
        self.model = RandomForest(self.types)
        self.acquisition_func = EI(self.model, compute_incumbent)
        self.local_search = LocalSearch(self.acquisition_func, self.config_space)
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

        # Aaron: I assume in RoBO always 2 dimensional numpy arrays (that is
        # how vectors are handled in GPy as well)
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
        #self.acquisition_func.update(self.model)

        found_configs = []
        acq_vals = np.array([n_iters])
        
        # Start N local search from different random start points  
        for i in range(n_iters):
            start_point = self.config_space.get_random_config_vector()
            configuration, acq_val = self.local_search.maximize(start_point)
            found_configs.append(configuration)
            acq_vals[i] = acq_val[0][0]
            
        
        # Return configuration with highest acquisition value
        best = np.argmax(acq_vals)
        return found_configs[best]
