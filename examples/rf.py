# TODO: remove really ugly boilerplate
import logging
import os
import inspect

import numpy as np
from sklearn.cross_validation import KFold
from pyrfr import regression32 as regression

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

def rfr(cfg, seed):
    """
    We optimize our own random forest with SMAC 
    """

    types=[0,3,0,0,20,6,3,2,0,0,0,20,0,7,0,0,0,20,2,0,0,2,0,20,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    types = np.array(types, dtype=np.uint)

    rf = regression.binary_rss()
    rf.num_trees = cfg["num_trees"]
    rf.seed = 42
    rf.do_bootstrapping = cfg["do_bootstrapping"] == "true"
    rf.num_data_points_per_tree = int(X.shape[0] * (3/4) * cfg["frac_points_per_tree"])
    rf.max_features = int(types.shape[0] * cfg["ratio_features"])
    rf.min_samples_to_split = cfg["min_samples_to_split"]
    rf.min_samples_in_leaf = cfg["min_samples_in_leaf"]
    rf.max_depth = cfg["max_depth"]
    #rf.epsilon_purity = cfg["eps_purity"]
    rf.max_num_nodes = cfg["max_num_nodes"]
    
    rmses = []
    for train, test in kf:
        X_train = X[train,:]
        y_train = y[train]
        X_test = X[test,:]
        y_test = y[test]
        
        data = regression.numpy_data_container(X_train,
                                                     y_train,
                                                     types)
        rf.fit(data)
        
        y_pred = []
        for x in X_test:
            y_p = rf.predict(x)[0]
            y_pred.append(y_p)
        y_pred = np.array(y_pred)
        
        rmse = np.sqrt(np.mean((y_pred - y_test)**2))
        rmses.append(rmse)
        
        #print(np.mean(rmses))
    return np.mean(rmses) 
    

logger = logging.getLogger("Optimizer") # Enable to show Debug outputs
logging.basicConfig(level=logging.INFO)

folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))

# load data
X = np.array(np.loadtxt(os.path.join(folder,"data/X.csv")), dtype=np.float32)
y = np.array(np.loadtxt(os.path.join(folder,"data/y.csv")), dtype=np.float32)

# cv folds
kf = KFold(X.shape[0], n_folds=4)

# build Configuration Space which defines all parameters and their ranges
# to illustrate different parameter types,
# we use continuous, integer and categorical parameters
cs = ConfigurationSpace()
do_bootstrapping = CategoricalHyperparameter(
    "do_bootstrapping", ["true","false"], default="true")
cs.add_hyperparameter(do_bootstrapping)
 
num_trees = UniformIntegerHyperparameter("num_trees", 10, 50, default=10)
cs.add_hyperparameter(num_trees)
 
frac_points_per_tree = UniformFloatHyperparameter("frac_points_per_tree", 0.001, 1, default=1)
cs.add_hyperparameter(frac_points_per_tree)
 
ratio_features = UniformFloatHyperparameter("ratio_features", 0.001, 1, default=1)
cs.add_hyperparameter(ratio_features)
 
min_samples_to_split = UniformIntegerHyperparameter("min_samples_to_split", 2, 20, default=2)
cs.add_hyperparameter(min_samples_to_split)
 
min_samples_in_leaf = UniformIntegerHyperparameter("min_samples_in_leaf", 1, 20, default=1)
cs.add_hyperparameter(min_samples_in_leaf)
 
max_depth = UniformIntegerHyperparameter("max_depth", 20, 100, default=20)
cs.add_hyperparameter(max_depth)
 
max_num_nodes = UniformIntegerHyperparameter("max_num_nodes", 100, 100000, default=1000)
cs.add_hyperparameter(max_num_nodes)
 
# SMAC scenario oject
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative runtime)
                     "runcount-limit": 400,  # at most 200 function evaluations
                     "cs": cs, # configuration space
                     "deterministic": "true",
                     "memory_limit": 1024,
                     })

# Optimize
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
            tae_runner=rfr)

# example call of the function
# it returns: Status, Cost, Runtime, Additional Infos
def_value = smac.solver.intensifier.tae_runner.run(
    cs.get_default_configuration(), 1)[1]
print("Default Value: %.2f" % (def_value))

try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = smac.solver.intensifier.tae_runner.run(incumbent, 1)[1]
print("Optimized Value: %.2f" % (inc_value))
