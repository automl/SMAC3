"""
An example for the usage of SMAC within Python.
We optimize a simple Random Forest on multiple user datasets (loaded from a file).

multiple instances; quality objective; no cutoff limits
"""


import logging
import os

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

INSTANCE_PATH = 'data/instances/'


def load_data_from_file(container_path: str, delimiter=','):
    """
    Loads dataset from the given folder. The datasets should be of the form - X.csv and y.csv
    :param container_path: folder where we can find X and y files
    :return: 2 numpy arrays containing X and y
    """
    X = np.loadtxt(container_path+'/X.csv', delimiter=delimiter)
    y = np.loadtxt(container_path+'/y.csv', delimiter=delimiter)
    return X, y


# Target Algorithm
def rf_from_cfg(cfg, instance, seed, **kwargs):
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

    # load dataset
    X, y = load_data_from_file(instance, delimiter=' ')

    def rmse(y, y_pred):
        return np.sqrt(np.mean((y_pred - y)**2))
    # Creating root mean square error for sklearns crossvalidation
    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    score = cross_val_score(rfr, X, y, cv=5, scoring=rmse_scorer)
    return -1 * np.mean(score)  # Because cross_validation sign-flips the score


def eval_on_instances(config, smac, instances):
    """
    Returns the mean cost over all instances
    :param smac: SMAC object, with a Target Algorithm executor
    :param instances: list of list of instances
    :return: averge cost
    """
    cost = 0.0
    for inst in instances:
        # Example call of the function with default values
        # It returns: Status, Cost, Runtime, Additional Infos
        cost += smac.get_tae_runner().run(config, inst[0])[1]

    return cost / len(instances)


# Load dataset
# boston = load_boston()
# If using multiple instances, they are normally sent to SMAC through the scenario object as
# filenames/folder names, which can be loaded from inside the target algorithm
instances = os.listdir(INSTANCE_PATH)
instances = [[INSTANCE_PATH+i] for i in instances]
sample_X, sample_y = load_data_from_file(instances[0][0], delimiter=' ')

logger = logging.getLogger("RF-example")
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)  # Enable to show debug-output
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
max_features = UniformIntegerHyperparameter("max_features", 1, sample_X.shape[1], default_value=1)
min_weight_frac_leaf = UniformFloatHyperparameter("min_weight_frac_leaf", 0.0, 0.5, default_value=0.0)
criterion = CategoricalHyperparameter("criterion", ["mse", "mae"], default_value="mse")
min_samples_to_split = UniformIntegerHyperparameter("min_samples_to_split", 2, 20, default_value=2)
min_samples_in_leaf = UniformIntegerHyperparameter("min_samples_in_leaf", 1, 20, default_value=1)
max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 10, 1000, default_value=100)

cs.add_hyperparameters([num_trees, min_weight_frac_leaf, criterion,
        max_features, min_samples_to_split, min_samples_in_leaf, max_leaf_nodes])

# SMAC scenario object
scenario = Scenario({"run_obj": "quality",     # we optimize quality (alternative runtime)
                     "wallclock-limit": 100,   # max duration to run the optimization (in seconds)
                     "cs": cs,                 # configuration space
                     "deterministic": "true",
                     "memory_limit": 3072,     # adapt this to reasonable value for your hardware
                     "instances": instances    # instances to load for training
                     })

# To optimize, we pass the function to the SMAC-object
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=rf_from_cfg)

def_value = eval_on_instances(cs.get_default_configuration(), smac, instances)
print("Value for default configuration: %.4f" % def_value)

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = eval_on_instances(incumbent, smac, instances)
print("Optimized Value: %.4f" % inc_value)
