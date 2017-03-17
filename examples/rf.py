# TODO: remove really ugly boilerplate
import logging
import os
import inspect

import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

def rfr(cfg, seed):
    """
    We optimize a random forest regressor with SMAC
    """
    rf = RandomForestRegressor(
            n_estimators=cfg["num_trees"],
            criterion=cfg["criterion"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_to_split"],
            min_samples_leaf=cfg["min_samples_in_leaf"],
            min_weight_fraction_leaf=cfg["min_weight_frac_leaf"],
            max_features=cfg["max_features"],
            max_leaf_nodes=cfg["max_leaf_nodes"],
            bootstrap=cfg["do_bootstrapping"],
            random_state=42)



    rmses = []
    for train, test in kf:
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        rmse = np.sqrt(np.mean((y_pred - y_test)**2))
        rmses.append(rmse)

        #print(np.mean(rmses))
    return np.mean(rmses)


logger = logging.getLogger("RF-example") # Enable to show Debug outputs
logging.basicConfig(level=logging.INFO)

folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))

# load data
X = np.array(np.loadtxt(os.path.join(folder,"data/X.csv")), dtype=np.float32)
y = np.array(np.loadtxt(os.path.join(folder,"data/y.csv")), dtype=np.float32)

# create cross-validation folds
kf = KFold(n_splits=4, shuffle=True, random_state=42)
kf = kf.split(X, y)

# build Configuration Space which defines all parameters and their ranges
# to illustrate different parameter types,
# we use continuous, integer and categorical parameters
cs = ConfigurationSpace()
do_bootstrapping = CategoricalHyperparameter(
    "do_bootstrapping", ["true","false"], default="true")
cs.add_hyperparameter(do_bootstrapping)

num_trees = UniformIntegerHyperparameter("num_trees", 10, 50, default=10)
cs.add_hyperparameter(num_trees)

criterion = CategoricalHyperparameter("criterion", ["mse","mae"], default="mse")
cs.add_hyperparameter(criterion)

max_depth = UniformIntegerHyperparameter("max_depth", 20, 30, default=20)
cs.add_hyperparameter(max_depth)

min_weight_frac_leaf=UniformFloatHyperparameter("min_weight_frac_leaf", 0.0, 0.5, default=0.0)
cs.add_hyperparameter(min_weight_frac_leaf)

max_features = UniformIntegerHyperparameter("max_features", 1, X.shape[1], default=1)
cs.add_hyperparameter(max_features)

min_samples_to_split = UniformIntegerHyperparameter("min_samples_to_split", 2, 20, default=2)
cs.add_hyperparameter(min_samples_to_split)

min_samples_in_leaf = UniformIntegerHyperparameter("min_samples_in_leaf", 1, 20, default=1)
cs.add_hyperparameter(min_samples_in_leaf)

max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 10, 1000, default=100)
cs.add_hyperparameter(max_leaf_nodes)

# SMAC scenario oject
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative runtime)
                     "runcount-limit": 50,  # at most 50 function evaluations
                     "cs": cs, # configuration space
                     "deterministic": "true",
                     "memory_limit": 1024,
                     })

# Optimize
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
            tae_runner=rfr)

# example call of the function
# it returns: Status, Cost, Runtime, Additional Infos
def_value = smac.get_tae_runner().run(cs.get_default_configuration(), 1)[1]
print("Value for default configuration: %.2f" % (def_value))

try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
print("Optimized Value: %.2f" % (inc_value))
