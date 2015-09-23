import logging

from robo.acquisition.ei import EI
from robo.maximizers.local_search import LocalSearch
from robo.solver.base_solver import BaseSolver


class BayesianOptimization(BaseSolver):
    # Implements the Bayesian optimization loop. I derived this class from the RoBO interface in order to share the output and logging functionality between RoBO and SMAC.
    def __init__(self):
        pass

    def run(self):
        # Here will be the main Bayesian optimization loop by iteratively calling the choose_next method
        pass

    def choose_next(self, X=None, Y=None):
        # Chooses the next configuration by training the model and optimizing the acquisition function. It performs basically one iteration of BO
        pass
