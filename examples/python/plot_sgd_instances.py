"""
SGD on Instances
^^^^^^^^^^^^^^^^

Example for optimizing a Multi-Layer Perceptron (MLP) using multiple instances.

Alternative to budgets, here we consider instances as a fidelity type. An instance represents a specific
scenario/condition (e.g. different datasets, subsets, transformations) for the algorithm to run. SMAC then returns the
algorithm that had the best performance across all the instances. In this case, an instance is a binary dataset i.e.,
digit-2 vs digit-3.

If we use instance as our fidelity, we need to initialize scenario with argument instance. In this case the argument
budget is no longer required by the target function.
"""

import logging

logging.basicConfig(level=logging.INFO)

import itertools
import warnings

import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF

# Import SMAC-utilities
from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


# We load the MNIST-dataset (a widely used benchmark) and split it into a list of binary datasets
digits = datasets.load_digits()
instances = [[str(a) + str(b)] for a, b in itertools.combinations(digits.target_names, 2)]


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


# Target Algorithm
def sgd_from_cfg(cfg, seed, instance):
    """Creates a SGD classifier based on a configuration and evaluates it on the
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
        A crossvalidated mean score for the SGD classifier on the loaded data-set.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # SGD classifier using given configuration
        clf = SGDClassifier(
            loss="log",
            penalty="elasticnet",
            alpha=cfg["alpha"],
            l1_ratio=cfg["l1_ratio"],
            learning_rate=cfg["learning_rate"],
            eta0=cfg["eta0"],
            max_iter=30,
            early_stopping=True,
            random_state=seed,
        )

        # get instance
        data, target = generate_instances(int(instance[0]), int(instance[1]))

        cv = StratifiedKFold(n_splits=4, random_state=seed, shuffle=True)  # to make CV splits consistent
        scores = cross_val_score(clf, data, target, cv=cv)

    return 1 - np.mean(scores)


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()

    # We define a few possible parameters for the SGD classifier
    alpha = UniformFloatHyperparameter("alpha", 0, 1, default_value=1.0)
    l1_ratio = UniformFloatHyperparameter("l1_ratio", 0, 1, default_value=0.5)
    learning_rate = CategoricalHyperparameter(
        "learning_rate", choices=["constant", "invscaling", "adaptive"], default_value="constant"
    )
    eta0 = UniformFloatHyperparameter("eta0", 0.00001, 1, default_value=0.1, log=True)
    # Add the parameters to configuration space
    cs.add_hyperparameters([alpha, l1_ratio, learning_rate, eta0])

    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "wallclock-limit": 100,  # max duration to run the optimization (in seconds)
            "cs": cs,  # configuration space
            "deterministic": True,
            "limit_resources": True,  # Uses pynisher to limit memory and runtime
            "memory_limit": 3072,  # adapt this to reasonable value for your hardware
            "cutoff": 3,  # runtime limit for the target algorithm
            "instances": instances,  # Optimize across all given instances
        }
    )

    # intensifier parameters
    # if no argument provided for budgets, hyperband decides them based on the number of instances available
    intensifier_kwargs = {
        "initial_budget": 1,
        "max_budget": 45,
        "eta": 3,
        # You can also shuffle the order of using instances by this parameter.
        # 'shuffle' will shuffle instances before each SH run and 'shuffle_once'
        # will shuffle instances once before the 1st SH iteration begins
        "instance_order": None,
    }

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=sgd_from_cfg,
        intensifier_kwargs=intensifier_kwargs,
    )

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
