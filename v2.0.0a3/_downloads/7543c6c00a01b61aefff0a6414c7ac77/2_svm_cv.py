"""
Support Vector Machine with Cross-Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of optimizing a simple support vector machine on the IRIS dataset. We use the
hyperparameter optimization facade, which uses a random forest as its surrogate model. It is able to
scale to higher evaluation budgets and a higher number of dimensions. Also, you can use mixed data
types as well as conditional hyperparameters.
"""

import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from ConfigSpace.conditions import InCondition
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score

from smac import HyperparameterOptimizationFacade, Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
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
        cs.add_hyperparameters([kernel, C, shrinking, degree, coef, gamma, gamma_value])
        cs.add_conditions([use_degree, use_coef, use_gamma, use_gamma_value])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        config_dict = config.get_dictionary()
        if "gamma" in config:
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
        n_trials=50,  # We want to run max 50 trials (combination of config and seed)
    )

    # We want to run the facade's default initial design, but we want to change the number
    # of initial configs to 5.
    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)

    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(
        scenario,
        classifier.train,
        initial_design=initial_design,
        overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
    )

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(classifier.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
