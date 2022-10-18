"""
SVM with Cross-Validation
^^^^^^^^^^^^^^^^^^^^^^^^^

An example to optimize a simple SVM on the IRIS-benchmark. SMAC4HPO is designed
for hyperparameter optimization (HPO) problems and uses an RF as its surrogate model.
It is able to scale to higher evaluation budgets and higher number of
dimensions. Also, you can use mixed data types as well as conditional hyperparameters.

SMAC4HPO by default only contains single fidelity approach. Therefore, only the configuration is
processed by the :term:`TAE`.
"""

import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score

from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


# We load the iris-dataset (a widely used benchmark)
iris = datasets.load_iris()


def svm_from_cfg(cfg):
    """Creates a SVM based on a configuration and evaluates it on the
    iris-dataset using cross-validation. Note here random seed is fixed

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    Returns:
    --------
    A crossvalidated mean score for the svm on the loaded data-set.
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"

    clf = svm.SVC(**cfg, random_state=42)

    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores)  # Minimize!


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()

    # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
    kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
    cs.add_hyperparameter(kernel)

    # There are some hyperparameters shared by all kernels
    C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0, log=True)
    shrinking = CategoricalHyperparameter("shrinking", [True, False], default_value=True)
    cs.add_hyperparameters([C, shrinking])

    # Others are kernel-specific, so we can add conditions to limit the searchspace
    degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)  # Only used by kernel poly
    coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
    cs.add_hyperparameters([degree, coef0])

    use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
    use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
    cs.add_conditions([use_degree, use_coef0])

    # This also works for parameters that are a mix of categorical and values
    # from a range of numbers
    # For example, gamma can be either "auto" or a fixed float
    gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
    gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default_value=1, log=True)
    cs.add_hyperparameters([gamma, gamma_value])
    # We only activate gamma_value if gamma is set to "value"
    cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
    # And again we can restrict the use of gamma in general to the choice of the kernel
    cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))

    # Scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount-limit": 50,  # max. number of function evaluations
            "cs": cs,  # configuration space
            "deterministic": True,
        }
    )

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = svm_from_cfg(cs.get_default_configuration())
    print("Default Value: %.2f" % (def_value))

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42), tae_runner=svm_from_cfg)

    incumbent = smac.optimize()

    inc_value = svm_from_cfg(incumbent)
    print("Optimized Value: %.2f" % (inc_value))

    # We can also validate our results (though this makes a lot more sense with instances)
    smac.validate(
        config_mode="inc",  # We can choose which configurations to evaluate
        # instance_mode='train+test',  # Defines what instances to validate
        repetitions=100,  # Ignored, unless you set "deterministic" to "false" in line 95
        n_jobs=1,
    )  # How many cores to use in parallel for optimization
