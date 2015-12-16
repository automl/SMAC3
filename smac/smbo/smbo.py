import logging
import numpy as np

from ConfigSpace.io import pcs
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant

from robo.acquisition.ei import EI
from robo.solver.base_solver import BaseSolver

from smac.smbo.rf_with_instances import RandomForestWithInstances
from smac.smbo.local_search import LocalSearch
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM


class SMBO(BaseSolver):

    def __init__(self, scenario, seed=42):
        '''
        Interface that contains the main Bayesian optimization loop

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object 
        seed: int
            Seed that is passed to random forest
        '''
        
        self.scenario = scenario
        self.config_space = scenario.cs

        # Extract types vector for rf from config space
        self.types = np.zeros(len(self.config_space.get_hyperparameters()))

        # Extract bounds of the input space
        X_lower = np.zeros([self.types.shape[0]])
        X_upper = np.zeros([self.types.shape[0]])

        for i, param in enumerate(self.config_space.get_hyperparameters()):
            if isinstance(param, (UniformFloatHyperparameter,
                                  UniformIntegerHyperparameter)):
                X_lower[i] = 0
                X_upper[i] = 1
            elif isinstance(param, (CategoricalHyperparameter)):
                n_cats = len(param.choices)
                self.types[i] = n_cats
                X_lower[i] = 0
                X_upper[i] = n_cats
            else:
                raise TypeError("Unknown hyperparameter type %s" % type(param))

        self.model = RandomForestWithInstances(self.types,
                                               scenario.feature_array)

        self.acquisition_func = EI(self.model,
                                   X_lower,
                                   X_upper)

        self.local_search = LocalSearch(self.acquisition_func,
                                        self.config_space)
        self.seed = seed

    def run(self, max_iters=10):
        '''
        Runs the Bayesian optimization loop for max_iters iterations

        Parameters
        ----------
        max_iters: int
            The maximum number of iterations

        Returns
        ----------
        incumbent: np.array(1, H)
            The best found configuration
        '''

        #TODO: Call initial design
        
        

        # Main BO loop
        self.runhistory = RunHistory()
        rh2EPM = RunHistory2EPM()
        for i in range(max_iters):

            #TODO: Transform lambda to X
            
            X, Y = rh2EPM.transform(self.runhistory)

            #TODO: Estimate new configuration
            next_config = self.choose_next(X, Y)

            #TODO: Perform intensification
            #self.incumbent = intensify(self.incumbent, next_config)

            #TODO: Perform target algorithm run

            #TODO: Update run history

            #TODO: Write run history into database

        return self.incumbent

    def choose_next(self, X=None, Y=None, n_iters=10):
        """
        Performs one single iteration of Bayesian optimization and estimated
        the next point to evaluate.

        Parameters
        ----------
        X : (N, D) numpy array, optional
            Each column contains a configuration and one set of instance features.
        Y : (N, 1) numpy array, optional
            The function values for each configuration instance pair.

        Returns
        -------
        x : (1, H) numpy array
            The suggested configuration to evaluate.
        """

        self.model.train(X, Y)
        self.acquisition_func.update(self.model)

        found_configs = []
        acq_vals = np.zeros([n_iters])

        # Start N local search from different random start points
        for i in range(n_iters):
            start_point = self.config_space.sample_configuration()
            configuration, acq_val = self.local_search.maximize(start_point)

            found_configs.append(configuration)
            acq_vals[i] = acq_val[0][0]

        # Return configuration with highest acquisition value
        best = np.argmax(acq_vals)
        #TODO: We could also return a configuration object here, but then also
        # the unit test has to be adapted
        return found_configs[best].get_array()[np.newaxis, :]
