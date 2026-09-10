"""Support Vector Machine with HPI-Based Random Search
# Flags: doc-Runnable

An example of optimizing a simple support vector machine on the IRIS dataset (see also
[Support Vector Machine with Cross-Validation](2_svm_cv.md)), using ``HPIRandomSearch`` as the acquisition
maximizer. ``HPIRandomSearch`` dynamically estimates via HyperSHAP which hyperparameters matter most, and restricts the search to those. This works for
single-objective optimization and for the multi-objective case (see [HPI-ParEGO](../3%20Multi-Objective/4_hpi_parego.md)).
"""

import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from ConfigSpace.conditions import InCondition
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score

from smac import HyperparameterOptimizationFacade, Scenario
from smac.acquisition.maximizer import HPIRandomSearch

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


# We load the iris-dataset (a widely used benchmark)
iris = datasets.load_iris()


class SVM:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace(seed=0)

        # First we create our hyperparameters
        kernel = Categorical("kernel", ["linear", "poly", "rbf", "sigmoid"], default="poly")
        C = Float("C", (0.001, 1000.0), default=1.0, log=True)
        shrinking = Categorical("shrinking", [True, False], default=True)
        degree = Integer("degree", (1, 5), default=3)
        coef = Float("coef0", (0.0, 10.0), default=0.0)
        gamma = Categorical("gamma", ["auto", "value"], default="auto")
        gamma_value = Float("gamma_value", (0.0001, 8.0), default=1.0, log=True)

        # Then we create dependencies
        use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
        use_coef = InCondition(child=coef, parent=kernel, values=["poly", "sigmoid"])
        use_gamma = InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"])
        use_gamma_value = InCondition(child=gamma_value, parent=gamma, values=["value"])

        # Add hyperparameters and conditions to our configspace
        cs.add([kernel, C, shrinking, degree, coef, gamma, gamma_value])
        cs.add([use_degree, use_coef, use_gamma, use_gamma_value])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        config_dict = dict(config)
        if "gamma" in config_dict:
            config_dict["gamma"] = config_dict["gamma_value"] if config_dict["gamma"] == "value" else "auto"
            config_dict.pop("gamma_value", None)

        classifier = svm.SVC(**config_dict, random_state=seed)
        scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
        cost = 1 - np.mean(scores)

        return cost


if __name__ == "__main__":
    classifier = SVM()

    # Next, we create an object, holding general information about the run
    scenario = Scenario(
        classifier.configspace,
        n_trials=100, 
    )

    # We want to run the facade's default initial design, but we want to change the number
    # of initial configs to 5.
    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)

    # HPIRandomSearch replaces the default acquisition maximizer.
    acquisition_maximizer = HPIRandomSearch(
        scenario.configspace, n_trials=scenario.n_trials
    )

    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(
        scenario,
        classifier.train,
        initial_design=initial_design,
        acquisition_maximizer=acquisition_maximizer,
        overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
    )

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(classifier.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
