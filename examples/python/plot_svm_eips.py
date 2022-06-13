"""
SVM with EIPS as acquisition functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example to optimize a simple SVM on the IRIS-benchmark with EIPS (EI per seconds)
acquisition function. Since EIPS requires two types of objections: EI values and the predicted
time used for the configurations. We need to fit the data
with a multi-objective model
"""

import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score

from smac.configspace import ConfigurationSpace
from smac.epm.uncorrelated_mo_rf_with_instances import (
    UncorrelatedMultiObjectiveRandomForestWithInstances,
)
from smac.facade.smac_ac_facade import SMAC4AC

# EIPS related
from smac.optimizer.acquisition import EIPS
from smac.runhistory.runhistory2epm import RunHistory2EPM4EIPS

# Import SMAC-utilities
from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

iris = datasets.load_iris()


# Target Algorithm
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
    print("Default Value: %.2f" % def_value)

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")

    # Besides the kwargs used for initializing UncorrelatedMultiObjectiveRandomForestWithInstances,
    # we also need kwargs for initializing the model insides UncorrelatedMultiObjectiveModel
    model_kwargs = {"target_names": ["loss", "time"], "model_kwargs": {"seed": 1}}
    smac = SMAC4AC(
        scenario=scenario,
        model=UncorrelatedMultiObjectiveRandomForestWithInstances,
        rng=np.random.RandomState(42),
        model_kwargs=model_kwargs,
        tae_runner=svm_from_cfg,
        acquisition_function=EIPS,
        runhistory2epm=RunHistory2EPM4EIPS,
    )

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
