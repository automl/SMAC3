"""
Speeding up Cross-Validation with Intensification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of optimizing a simple support vector machine on the digits dataset. In contrast to the
[simple example](examples/1_basics/2_svm_cv.py), in which all cross-validation folds are executed
at once, we use the intensification mechanism described in the original 
[SMAC paper](https://link.springer.com/chapter/10.1007/978-3-642-25566-3_40) as also demonstrated
by [Auto-WEKA](https://dl.acm.org/doi/10.1145/2487575.2487629). This mechanism allows us to
terminate the evaluation of a configuration if after a certain number of folds, the configuration
is found to be worse than the incumbent configuration. This is especially useful if the evaluation
of a configuration is expensive, e.g., if we have to train a neural network or if we have to
evaluate the configuration on a large dataset.
"""
__copyright__ = "Copyright 2023, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

N_FOLDS = 10  # Global variable that determines the number of folds

from ConfigSpace import Configuration, ConfigurationSpace, Float
from sklearn import datasets, svm
from sklearn.model_selection import StratifiedKFold

from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier import Intensifier

# We load the digits dataset, a small-scale 10-class digit recognition dataset
X, y = datasets.load_digits(return_X_y=True)


class SVM:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace(seed=0)

        # First we create our hyperparameters
        C = Float("C", (2 ** - 5, 2 ** 15), default=1.0, log=True)
        gamma = Float("gamma", (2 ** -15, 2 ** 3), default=1.0, log=True)

        # Add hyperparameters to our configspace
        cs.add_hyperparameters([C, gamma])

        return cs

    def train(self, config: Configuration, instance: str, seed: int = 0) -> float:
        """Creates a SVM based on a configuration and evaluate on the given fold of the digits dataset
        
        Parameters
        ----------
        config: Configuration
            The configuration to train the SVM.
        instance: str
            The name of the instance this configuration should be evaluated on. This is always of type
            string by definition. In our case we cast to int, but this could also be the filename of a
            problem instance to be loaded.
        seed: int
            The seed used for this call.
        """
        instance = int(instance)
        classifier = svm.SVC(**config, random_state=seed)
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (train_idx, test_idx) in enumerate(splitter.split(X=X, y=y)):
            if k != instance:
                continue
            else:
                train_X = X[train_idx]
                train_y = y[train_idx]
                test_X = X[test_idx]
                test_y = y[test_idx]
                classifier.fit(train_X, train_y)
                cost = 1 - classifier.score(test_X, test_y)

        return cost


if __name__ == "__main__":
    classifier = SVM()

    # Next, we create an object, holding general information about the run
    scenario = Scenario(
        classifier.configspace,
        n_trials=50,  # We want to run max 50 trials (combination of config and instances in the case of
                      # deterministic=True. In the case of deterministic=False, this would be the
                      # combination of instances, seeds and configs). The number of distinct configurations
                      # evaluated by SMAC will be lower than this number because some of the configurations
                      # will be executed on more than one instance (CV fold).
        instances=[f"{i}" for i in range(N_FOLDS)],  # Specify all instances by their name (as a string)
        instance_features={f"{i}": [i] for i in range(N_FOLDS)}, # breaks SMAC
        deterministic=True  # To simplify the problem we make SMAC believe that we have a deterministic
                            # optimization problem.
        
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
        # The next line defines the intensifier, i.e., the module that governs the selection of 
        # instance-seed pairs. Since we set deterministic to True above, it only governs the instance in
        # this example. Technically, it is not necessary to create the intensifier as a user, but it is
        # necessary to do so because we change the argument max_config_calls (the number of instance-seed pairs
        # per configuration to try) to the number of cross-validation folds, while the default would be 3.
        intensifier=Intensifier(scenario=scenario, max_config_calls=N_FOLDS, seed=0)
    )

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(classifier.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")

    # Let's see how many configurations we have evaluated. If this number is higher than 5, we have looked
    # at more configurations than would have been possible with regular cross-validation, where the number
    # of configurations would be determined by the number of trials divided by the number of folds (50 / 10).
    runhistory = smac.runhistory
    print(f"Number of evaluated configurations: {len(runhistory.config_ids)}")