import logging

# I already implemented the local search in ESMAC and EI in RoBO maybe we can reuse it here
from robo.acquisition.ei import EI
from robo.maximizers.local_search import LocalSearch
from robo.solver.base_solver import BaseSolver

import smac.smbo.intensification
import smac.smbo.run_history


class BayesianOptimization(BaseSolver):
    # Implements the Bayesian optimization loop. I derived this class from the RoBO interface in order to share the output and logging functionality between RoBO and SMAC.
    def __init__(self):
        self.model = RandomForest()
        self.acquisition_func = EI(self.model)
        self.local_search = LocalSearch(self.acquisition_func)

    def run(self, max_iters):
        # Here will be the main Bayesian optimization loop by iteratively calling the choose_next method
        #TODO: Initialize X, Y
        X = None
        Y = None
        for i in range(max_iters):
            next_config = self.choose_next(X, Y)
            incumbent = itensify()
            #TODO: Evaluate nex_config and update X, Y
        return incumbent

    def choose_next(self, X=None, Y=None):
        # Chooses the next configuration by training the model and optimizing the acquisition function. It performs basically one iteration of BO
        self.model.fit(X, Y)
        self.acquisition_func.update(self.model)
        configuration = self.local_search.maximize()
        return configuration
