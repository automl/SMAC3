"""
An example for the usage of SMAC within Python.
We optimize a SVM on the MNIST digits dataset as multiple binary classification problems
using "Hyperband" intensification. We split the MNIST digits dataset (10 classes) into 45 binary datasets.

In this example, we use instances as the budget in hyperband and optimize the average cross validation accuracy.
An "Instance" represents a specific scenario/condition (eg: different datasets, subsets, transformations)
for the algorithm to run. SMAC then returns the algorithm that had the best performance across all the instances.
In this case, an instance here is a binary dataset i.e., digit-2 vs digit-3.
"""

import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score, StratifiedKFold
import itertools

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.intensification.hyperband import Hyperband


# We load the MNIST-dataset (a widely used benchmark) and split it into a collection of binary datasets
digits = datasets.load_digits()
instances = [[str(a)+str(b)] for a, b in itertools.combinations(digits.target_names, 2)]


def generate_instances(a: int, b: int):
    """
    Function to select data for binary classification from the digits dataset
    a & b are the two classes
    """
    # get indices of both classes
    indices = np.where(np.logical_or(a == digits.target, b == digits.target))
    # get data
    data = digits.data[indices]
    target = digits.target[indices]
    return data, target


def svm_from_cfg(cfg, seed, instance):
    """ Creates a SVM based on a configuration and evaluates it on the
    digits dataset using cross-validation.

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!
    seed: int or RandomState
        used to initialize the svm's random generator
    instance: str
        used to represent the instance to use (the 2 classes to consider in this case)

    Returns:
    --------
    float
        A crossvalidated mean score for the svm on the loaded data-set.
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"

    clf = svm.SVC(**cfg, random_state=seed)

    # get instance
    data, target = generate_instances(int(instance[0]), int(instance[1]))

    cv = StratifiedKFold(n_splits=4, random_state=seed)  # to make CV splits consistent
    scores = cross_val_score(clf, data, target, cv=cv)
    return 1-np.mean(scores)  # Minimize!


logger = logging.getLogger("SVM-example")
logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

# We define a few possible types of SVM-kernels and add them as "kernel" to our cs
kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="rbf")
cs.add_hyperparameter(kernel)

# There are some hyperparameters shared by all kernels
C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default_value="false")
cs.add_hyperparameters([C, shrinking])

# Others are kernel-specific, so we can add conditions to limit the searchspace
degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)     # Only used by kernel poly
coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
cs.add_hyperparameters([degree, coef0])
use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
cs.add_conditions([use_degree, use_coef0])

# This also works for parameters that are a mix of categorical and values from a range of numbers
# For example, gamma can be either "auto" or a fixed float
gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="value")  # only rbf, poly, sigmoid
gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default_value=1)
cs.add_hyperparameters([gamma, gamma_value])
# We only activate gamma_value if gamma is set to "value"
cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
# And again we can restrict the use of gamma in general to the choice of the kernel
cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))

# SMAC scenario object
scenario = Scenario({"run_obj": "quality",      # we optimize quality (alternative to runtime)
                     "wallclock-limit": 60,     # max duration to run the optimization (in seconds)
                     "cs": cs,                  # configuration space
                     "deterministic": "true",
                     "limit_resources": True,   # Uses pynisher to limit memory and runtime
                     "memory_limit": 3072,      # adapt this to reasonable value for your hardware
                     "instances": instances     # Optimize across all given instances
                     })

# intensifier parameters
# if no argument provided for budgets, hyperband decides them based on the number of instances available
intensifier_kwargs = {'initial_budget': 1, 'max_budget': 45, 'eta': 3,
                      'instance_order': None,  # You can also shuffle the order of using instances by this parameter.
                                               # 'shuffle' will shuffle instances before each SH run and
                                               # 'shuffle_once' will shuffle instances once before the 1st
                                               # SH iteration begins
                      }

# To optimize, we pass the function to the SMAC-object
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=svm_from_cfg,
                intensifier=Hyperband,                  # you can also change the intensifier to use like this!
                                                        # This example currently uses Hyperband intensification,
                intensifier_kwargs=intensifier_kwargs)  # all parameters related to intensifier can be passed like this


# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_costs = []
for i in instances:
    cost = smac.get_tae_runner().run(cs.get_default_configuration(), i[0])[1]
    def_costs.append(cost)
print("Value for default configuration: %.4f" % (np.mean(def_costs)))

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_costs = []
for i in instances:
    cost = smac.get_tae_runner().run(incumbent, i[0])[1]
    inc_costs.append(cost)
print("Optimized Value: %.4f" % (np.mean(inc_costs)))
