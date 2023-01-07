"""
User Priors over the Optimum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example for optimizing a Multi-Layer Perceptron (MLP) setting priors over the optimum on the
hyperparameters. These priors are derived from user knowledge (from previous runs on similar
tasks, common knowledge or intuition gained from manual tuning). To create the priors, we make
use of the Normal and Beta Hyperparameters, as well as the "weights" property of the
``CategoricalHyperparameter``. This can be integrated into the optimiztion for any SMAC facade,
but we stick with the hyperparameter optimization facade here. To incorporate user priors into the
optimization, you have to change the acquisition function to ``PriorAcquisitionFunction``.
"""

import warnings

import numpy as np
from ConfigSpace import (
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    NormalFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from smac import HyperparameterOptimizationFacade, Scenario
from smac.acquisition.function import PriorAcquisitionFunction

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


digits = load_digits()


class MLP:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types,
        # we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace()

        # We do not have an educated belief on the number of layers beforehand
        # As such, the prior on the HP is uniform
        n_layer = UniformIntegerHyperparameter(
            "n_layer",
            lower=1,
            upper=5,
        )

        # We believe the optimal network is likely going to be relatively wide,
        # And place a Beta Prior skewed towards wider networks in log space
        n_neurons = BetaIntegerHyperparameter(
            "n_neurons",
            lower=8,
            upper=256,
            alpha=4,
            beta=2,
            log=True,
        )

        # We believe that ReLU is likely going to be the optimal activation function about
        # 60% of the time, and thus place weight on that accordingly
        activation = CategoricalHyperparameter(
            "activation",
            ["logistic", "tanh", "relu"],
            weights=[1, 1, 3],
            default_value="relu",
        )

        # Moreover, we believe ADAM is the most likely optimizer
        optimizer = CategoricalHyperparameter(
            "optimizer",
            ["sgd", "adam"],
            weights=[1, 2],
            default_value="adam",
        )

        # We do not have an educated opinion on the batch size, and thus leave it as-is
        batch_size = UniformIntegerHyperparameter(
            "batch_size",
            16,
            512,
            default_value=128,
        )

        # We place a log-normal prior on the learning rate, so that it is centered on 10^-3,
        # with one unit of standard deviation per multiple of 10 (in log space)
        learning_rate_init = NormalFloatHyperparameter(
            "learning_rate_init",
            lower=1e-5,
            upper=1.0,
            mu=np.log(1e-3),
            sigma=np.log(10),
            log=True,
        )

        # Add all hyperparameters at once:
        cs.add_hyperparameters([n_layer, n_neurons, activation, optimizer, batch_size, learning_rate_init])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            classifier = MLPClassifier(
                hidden_layer_sizes=[config["n_neurons"]] * config["n_layer"],
                solver=config["optimizer"],
                batch_size=config["batch_size"],
                activation=config["activation"],
                learning_rate_init=config["learning_rate_init"],
                random_state=seed,
                max_iter=5,
            )

            # Returns the 5-fold cross validation accuracy
            cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)  # to make CV splits consistent
            score = cross_val_score(classifier, digits.data, digits.target, cv=cv, error_score="raise")

        return 1 - np.mean(score)


if __name__ == "__main__":
    mlp = MLP()
    default_config = mlp.configspace.get_default_configuration()

    # Define our environment variables
    scenario = Scenario(mlp.configspace, n_trials=40)

    # We also want to include our default configuration in the initial design
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario,
        additional_configs=[default_config],
    )

    # We define the prior acquisition function, which conduct the optimization using priors over the optimum
    acquisition_function = PriorAcquisitionFunction(
        acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(scenario),
        decay_beta=scenario.n_trials / 10,  # Proven solid value
    )

    # We only want one config call (use only one seed in this example)
    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=1,
    )

    # Create our SMAC object and pass the scenario and the train method
    smac = HyperparameterOptimizationFacade(
        scenario,
        mlp.train,
        initial_design=initial_design,
        acquisition_function=acquisition_function,
        intensifier=intensifier,
        overwrite=True,
    )

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(default_config)
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Default cost: {incumbent_cost}")
