import logging
import numpy as np

# Aaron: I already implemented the local search in ESMAC and EI in RoBO maybe we can reuse it here
# from robo.acquisition.ei import EI
# from robo.maximizers.local_search import LocalSearch
# from robo.solver.base_solver import BaseSolver


class BayesianOptimization(object):
    # Aaron: I would like to derive this class from the RoBO interface in order to share the output and logging functionality between RoBO and SMAC.
    # Aaron: I will change the default value to RandomForest, EI and LocalSearch as soon as we have configured the dependencies
    def __init__(self, model=None, acquisition_func=None, maximizer=None, seed=42):
        '''
            Implementation of the main Bayesian optimization loop
            Args:
                model : A model that captures our believe of our objective function (robo model object)
                acquisition_func: Surrogate function to pick a new configuration (robo acquisition object)
                maximizer: Optimization strategy to maximize the acquisition function (robo maximizer object)
                seed : random seed (integer)
        '''
        self.model = model
        # Aaron: Maybe we do not want to pass the acquisition function and the maximizer as argument but rather initialize them here. Passing them
        # as an argument makes only sense if we want to use something different than EI and LocalSearch
        self.acquisition_func = acquisition_func
        self.maximizer = maximizer
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

        # Aaron: I assume in RoBO always 2 dimensional numpy arrays (that is how vectors are handled in GPy as well)
        incumbent = np.random.randn(1, 1)
        return incumbent

    def choose_next(self, X=None, Y=None):
        '''
            Chooses the next configuration by training the model and optimizing the acquisition function.
            Args:
                X : The configuration we have seen so far (2D numpy array)
                Y : The function values of the configurations (2D numpy array)
            Return:
                incumbent: The next configuration to evaluate (2D numpy array)
        '''
        #self.model.fit(X, Y)
        #self.acquisition_func.update(self.model)
        #configuration = self.local_search.maximize()
        configuration = np.random.randn(1, 1)
        return configuration
