#!/bin/python3

import numpy as np

from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType

from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost, RunHistory2EPM4Cost

from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.rfr_imputator import RFRImputator

from ConfigSpace.io import pcs
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant


def convert_data(scenario:Scenario, runhistory:RunHistory):
    '''
        converts data from runhistory into EPM format
        
        Parameters
        ----------
        scenario: Scenario
            smac.scenario.scenario.Scenario Object 
        runhistory: RunHistory
            smac.runhistory.runhistory.RunHistory Object with all necessary data
            
        Returns
        -------
        np.array
            X matrix with configuartion x features for all observed samples
        np.array
            y matrix with all observations
        np.array
            types of X cols -- necessary to train our RF implementation
    '''
    
    types = np.zeros(len(scenario.cs.get_hyperparameters()),
                         dtype=np.uint)

    for i, param in enumerate(scenario.cs.get_hyperparameters()):
        if isinstance(param, (CategoricalHyperparameter)):
            n_cats = len(param.choices)
            types[i] = n_cats

    if scenario.feature_array is not None:
        types = np.hstack(
            (types, np.zeros((scenario.feature_array.shape[1]))))

    types = np.array(types, dtype=np.uint)

    model = RandomForestWithInstances(types,
                                           scenario.feature_array)

    params = scenario.cs.get_hyperparameters()
    num_params = len(params)

    if scenario.run_obj == "runtime":
        if scenario.run_obj == "runtime":
            # if we log the performance data,
            # the RFRImputator will already get
            # log transform data from the runhistory
            cutoff = np.log10(scenario.cutoff)
            threshold = np.log10(scenario.cutoff *
                                 scenario.par_factor)
        else:
            cutoff = scenario.cutoff
            threshold = scenario.cutoff * scenario.par_factor

        imputor = RFRImputator(cs=scenario.cs,
                               rs=np.random.RandomState(42),
                               cutoff=cutoff,
                               threshold=threshold,
                               model=model,
                               change_threshold=0.01,
                               max_iter=10)
        # TODO: Adapt runhistory2EPM object based on scenario
        rh2EPM = RunHistory2EPM4LogCost(scenario=scenario,
                                        num_params=num_params,
                                        success_states=[
                                            StatusType.SUCCESS, ],
                                        impute_censored_data=False,
                                        impute_state=[
                                            StatusType.TIMEOUT, ],
                                        imputor=imputor)
    else:
        rh2EPM = RunHistory2EPM4Cost(scenario=scenario,
                                     num_params=num_params,
                                     success_states=None,
                                     impute_censored_data=False,
                                     impute_state=None)
    X, Y = rh2EPM.transform(runhistory)
    
    return X, Y, types