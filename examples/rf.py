import logging
import os
import inspect

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

boston = load_boston()

def rf_from_cfg(cfg, seed):
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
        min_samples_split=cfg["min_samples_to_split"],
        min_samples_leaf=cfg["min_samples_in_leaf"],
        min_weight_fraction_leaf=cfg["min_weight_frac_leaf"],
        max_features=cfg["max_features"],
        max_leaf_nodes=cfg["max_leaf_nodes"],
        bootstrap=cfg["do_bootstrapping"],
        random_state=seed)

    def rmse(y, y_pred):
        return np.sqrt(np.mean((y_pred - y)**2))
    # Creating root mean square error for sklearns crossvalidation
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    score = cross_val_score(rfr, boston.data, boston.target, cv=11, scoring=rmse_scorer)
    return -1 * np.mean(score)  # Because cross_validation sign-flips the score


logger = logging.getLogger("RF-example")
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)  # Enable to show debug-output
logger.info("Running random forest example for SMAC. If you experience "
            "difficulties, try to decrease the memory-limit.")

# Build Configuration Space which defines all parameters and their ranges.
# To illustrate different parameter types,
# we use continuous, integer and categorical parameters.
cs = ConfigurationSpace()

# We can add single hyperparameters:
do_bootstrapping = CategoricalHyperparameter(
    "do_bootstrapping", ["true", "false"], default_value="true")
cs.add_hyperparameter(do_bootstrapping)

# Or we can add multiple hyperparameters at once:
num_trees = UniformIntegerHyperparameter("num_trees", 10, 50, default_value=10)
max_features = UniformIntegerHyperparameter("max_features", 1, boston.data.shape[1], default_value=1)
min_weight_frac_leaf = UniformFloatHyperparameter("min_weight_frac_leaf", 0.0, 0.5, default_value=0.0)
criterion = CategoricalHyperparameter("criterion", ["mse", "mae"], default_value="mse")
min_samples_to_split = UniformIntegerHyperparameter("min_samples_to_split", 2, 20, default_value=2)
min_samples_in_leaf = UniformIntegerHyperparameter("min_samples_in_leaf", 1, 20, default_value=1)
max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 10, 1000, default_value=100)

cs.add_hyperparameters([num_trees, min_weight_frac_leaf, criterion,
        max_features, min_samples_to_split, min_samples_in_leaf, max_leaf_nodes])

# SMAC scenario oject
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternative runtime)
                     "runcount-limit": 50,  # maximum number of function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true",
                     "memory_limit": 3072,   # adapt this to reasonable value for your hardware
                     })

# To optimize, we pass the function to the SMAC-object
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
            tae_runner=rf_from_cfg)

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
