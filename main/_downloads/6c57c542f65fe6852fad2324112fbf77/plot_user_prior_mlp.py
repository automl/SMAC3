"""
HPO with User Priors over the Optimum
^^^^^^^^^^^^^^^^^^^^^^^

Example for optimizing a Multi-Layer Perceptron (MLP) setting priors over the optimum on the
hyperparameters. These priors are derived from user knowledge - from previous runs on similar
tasks, common knowledge or intuition gained from manual tuning. To create the priors, we make
use of the Normal and Beta Hyperparameters, as well as the "weights" property of the
CategoricalHyperparameter. This can be integrated into the optimiztion for any SMAC facade,
but we stick with SMAC4HPO here. To incorporate user priors into the optimization, 
πBO (nolinkexistsyet) is used to bias the point selection strategy.

MLP is used as the deep neural network.
The digits datasetis chosen to optimize the average accuracy on 5-fold cross validation.
"""

import logging

logging.basicConfig(level=logging.INFO)

import warnings

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import (
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    NormalFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from smac.configspace import ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


digits = load_digits()


# Target Algorithm
def mlp_from_cfg(cfg, seed):
    """
    Creates a MLP classifier from sklearn and fits the given data on it.

    Parameters
    ----------
    cfg: Configuration
        configuration chosen by smac
    seed: int or RandomState
        used to initialize the rf's random generator
    budget: float
        used to set max iterations for the MLP

    Returns
    -------
    float
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        mlp = MLPClassifier(
            hidden_layer_sizes=[cfg["n_neurons"]] * cfg["n_layer"],
            solver=cfg["optimizer"],
            batch_size=cfg["batch_size"],
            activation=cfg["activation"],
            learning_rate_init=cfg["learning_rate_init"],
            random_state=seed,
        )

        # returns the cross validation accuracy
        cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)  # to make CV splits consistent
        score = cross_val_score(mlp, digits.data, digits.target, cv=cv, error_score="raise")

    return 1 - np.mean(score)


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()

    # We do not have an educated belief on the number of layers beforehand
    # As such, the prior on the HP is uniform
    n_layer = UniformIntegerHyperparameter("n_layer", lower=1, upper=5)

    # We believe the optimal network is likely going to be relatively wide,
    # And place a Beta Prior skewed towards wider networks in log space
    n_neurons = BetaIntegerHyperparameter("n_neurons", lower=8, upper=1024, alpha=4, beta=2, log=True)

    # We believe that ReLU is likely going to be the optimal activation function about
    # 60% of the time, and thus place weight on that accordingly
    activation = CategoricalHyperparameter(
        "activation", ["logistic", "tanh", "relu"], weights=[1, 1, 3], default_value="relu"
    )

    # Moreover, we believe ADAM is the most likely optimizer
    optimizer = CategoricalHyperparameter("optimizer", ["sgd", "adam"], weights=[1, 2], default_value="adam")

    # We do not have an educated opinion on the batch size, and thus leave it as-is
    batch_size = UniformIntegerHyperparameter("batch_size", 16, 512, default_value=128)

    # We place a log-normal prior on the learning rate, so that it is centered on 10^-3,
    # with one unit of standard deviation per multiple of 10 (in log space)
    learning_rate_init = NormalFloatHyperparameter(
        "learning_rate_init", lower=1e-5, upper=1.0, mu=np.log(1e-3), sigma=np.log(10), log=True
    )

    # Add all hyperparameters at once:
    cs.add_hyperparameters([n_layer, n_neurons, activation, optimizer, batch_size, learning_rate_init])

    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "runcount-limit": 20,  # max duration to run the optimization (in seconds)
            "cs": cs,  # configuration space
            "deterministic": "true",
            "limit_resources": True,  # Uses pynisher to limit memory and runtime
            # Alternatively, you can also disable this.
            # Then you should handle runtime and memory yourself in the TA
            "cutoff": 30,  # runtime limit for target algorithm
            "memory_limit": 3072,  # adapt this to reasonable value for your hardware
        }
    )

    # The rate at which SMAC forgets the prior.
    # The higher the value, the more the prior is considered.
    # Defaults to # n_iterations / 10
    user_prior_kwargs = {"decay_beta": 1.5}

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4HPO(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=mlp_from_cfg,
        # This flag is required to conduct the optimisation using priors over the optimum
        user_priors=True,
        user_prior_kwargs=user_prior_kwargs,
        # Using random configurations will cause the initialization to be samples drawn from the prior
        initial_design=RandomConfigurations,
    )

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(), seed=0)[1]

    print("Value for default configuration: %.4f" % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(config=incumbent, seed=0)[1]

    print("Optimized Value: %.4f" % inc_value)
