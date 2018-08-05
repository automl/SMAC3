import logging
import os
import typing

import numpy as np

from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, \
    RandomConfigurationChooser, ChooserCosineAnnealing
from smac.facade.smac_facade import SMAC

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class BORF(SMAC):
    """
    Facade to use BORF default mode
    
    see smac.facade.smac_Facade for API

    Attributes
    ----------
    logger
    stats : Stats
    solver : SMBO
    runhistory : RunHistory
        List with information about previous runs
    trajectory : list
        List of all incumbents

    """

    def __init__(self,**kwargs):
        """
        Constructor
        see ~smac.facade.smac_facade for docu
        """
        
        super().__init__(**kwargs)
        
        #  RF settings
        self.solver.model.rf_opts.num_trees = 10
        self.solver.model.rf_opts.do_bootstrapping = True
        self.solver.model.rf_opts.tree_opts.max_features = self.solver.model.types.shape[0]
        self.solver.model.rf_opts.tree_opts.min_samples_to_split = 2
        self.solver.model.rf_opts.tree_opts.min_samples_in_leaf = 1
        
        # no random configurations
        rand_chooser = ChooserNoCoolDown(10**10)
        self.solver.random_configuration_chooser = rand_chooser
        
        # only 1 configuration per SMBO iteration
        self.solver.scenario.intensification_percentage = 1e-10
        self.solver.intensifier.min_chall = 1
        
        
    @staticmethod
    def _get_random_configuration_chooser(random_configuration_chooser:RandomConfigurationChooser, 
                                          rng:np.random.RandomState):
        """
        Initialize random configuration chooser
        If random_configuration_chooser is falsy, initialize with ChooserNoCoolDown(2.0)

        Parameters
        ----------
        random_configuration_chooser: RandomConfigurationChooser
            generator for picking random configurations
            or configurations optimized based on acquisition function
        rng : np.random.RandomState
            Random number generator

        Returns
        -------
        RandomConfigurationChooser

        """
        if not random_configuration_chooser:
            #return ChooserCosineAnnealing(prob_max=0.5, prob_min=0.001, 
            #     restart_iteration= 10,
            #     rng=rng)
            return ChooserNoCoolDown(2.0)
        return random_configuration_chooser

    