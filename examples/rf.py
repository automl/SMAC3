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
        Creates a random forest regressor from sklearn and fits the given data on it.
        This is the function-call we try to optimize. Chosen values are stored in
        the configuration (cfg).

        Parameters:
        -----------
        cfg: Configuration
            configuration chosen by smac
        seed: int or RandomState
            used to initialize the rf's random generator

        Returns:
        -----------
        np.mean(rmses): float
            mean of root mean square errors of random-forest test predictions
            per cv-fold
    """
    rfr = RandomForestRegressor(
        n_estimators=cfg["num_trees"],
        criterion=cfg["criterion"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_to_split"],
        min_samples_leaf=cfg["min_samples_in_leaf"],
        min_weight_fraction_leaf=cfg["min_weight_frac_leaf"],
        max_features=cfg["max_features"],
        max_leaf_nodes=cfg["max_leaf_nodes"],
        bootstrap=cfg["do_bootstrapping"],
        random_state=seed)

    rmses = []
    for train, test in kf:
        # We iterate over cv-folds
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        rfr.fit(X_train, y_train)

        y_pred = rfr.predict(X_test)

        # We use root mean square error as performance measure
        rmse = np.sqrt(np.mean((y_pred - y_test)**2))
        rmses.append(rmse)
    return np.mean(rmses)


logger = logging.getLogger("RF-example")
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)  # Enable to show debug-output

folder = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))

# Load data
X = np.array(np.loadtxt(os.path.join(folder, "data/X.csv")), dtype=np.float32)
y = np.array(np.loadtxt(os.path.join(folder, "data/y.csv")), dtype=np.float32)

# Create cross-validation folds
kf = KFold(n_splits=4, shuffle=True, random_state=42)
kf = kf.split(X, y)

# Build Configuration Space which defines all parameters and their ranges.
# To illustrate different parameter types,
# we use continuous, integer and categorical parameters.
cs = ConfigurationSpace()

# We can add single hyperparameters:
do_bootstrapping = CategoricalHyperparameter(
    "do_bootstrapping", ["true", "false"], default="true")
cs.add_hyperparameter(do_bootstrapping)

# Or we can add multiple hyperparameters at once:
num_trees = UniformIntegerHyperparameter("num_trees", 10, 50, default=10)
max_depth = UniformIntegerHyperparameter("max_depth", 20, 30, default=20)
max_features = UniformIntegerHyperparameter("max_features", 1, X.shape[1], default=1)
min_weight_frac_leaf = UniformFloatHyperparameter("min_weight_frac_leaf", 0.0, 0.5, default=0.0)
criterion = CategoricalHyperparameter("criterion", ["mse", "mae"], default="mse")
min_samples_to_split = UniformIntegerHyperparameter("min_samples_to_split", 2, 20, default=2)
min_samples_in_leaf = UniformIntegerHyperparameter("min_samples_in_leaf", 1, 20, default=1)
max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 10, 1000, default=100)

cs.add_hyperparameters([num_trees, max_depth, min_weight_frac_leaf, criterion,
        max_features, min_samples_to_split, min_samples_in_leaf, max_leaf_nodes])

# SMAC scenario oject
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative runtime)
                     "runcount-limit": 20,  # maximum number of function evaluations
                     "cs": cs,              # configuration space
                     "deterministic": "true",
                     "memory_limit": 1024,
                     })

# To optimize, we pass the function to the SMAC-object
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
            tae_runner=rfr)

# Example call of the function with default values
# It returns: Status, Cost, Runtime, Additional Infos
def_value = smac.get_tae_runner().run(cs.get_default_configuration(), 1)[1]
print("Value for default configuration: %.2f" % (def_value))

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
print("Optimized Value: %.2f" % (inc_value))
