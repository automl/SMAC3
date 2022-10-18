"""
MLP with Multi-Fidelity
^^^^^^^^^^^^^^^^^^^^^^^

Example for optimizing a Multi-Layer Perceptron (MLP) using multiple budgets.
Since we want to take advantage of Multi-Fidelity, the SMAC4MF facade is a good choice. By default,
SMAC4MF internally runs with `hyperband <https://arxiv.org/abs/1603.06560>`_, which is a combination of an
aggressive racing mechanism and successive halving.

MLP is a deep neural network, and therefore, we choose epochs as fidelity type. The digits dataset
is chosen to optimize the average accuracy on 5-fold cross validation.

.. note::

    This example uses the ``SMAC4MF`` facade, which is the closest implementation to
    `BOHB <https://github.com/automl/HpBandSter>`_.
"""

import logging

logging.basicConfig(level=logging.INFO)

import warnings

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


digits = load_digits()


# Target Algorithm
def mlp_from_cfg(cfg, seed, budget):
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

    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the MLP, so we replace them with placeholder values.
    lr = cfg["learning_rate"] if cfg["learning_rate"] else "constant"
    lr_init = cfg["learning_rate_init"] if cfg["learning_rate_init"] else 0.001
    batch_size = cfg["batch_size"] if cfg["batch_size"] else 200

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        mlp = MLPClassifier(
            hidden_layer_sizes=[cfg["n_neurons"]] * cfg["n_layer"],
            solver=cfg["solver"],
            batch_size=batch_size,
            activation=cfg["activation"],
            learning_rate=lr,
            learning_rate_init=lr_init,
            max_iter=int(np.ceil(budget)),
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

    n_layer = UniformIntegerHyperparameter("n_layer", 1, 5, default_value=1)
    n_neurons = UniformIntegerHyperparameter("n_neurons", 8, 1024, log=True, default_value=10)
    activation = CategoricalHyperparameter("activation", ["logistic", "tanh", "relu"], default_value="tanh")
    solver = CategoricalHyperparameter("solver", ["lbfgs", "sgd", "adam"], default_value="adam")
    batch_size = UniformIntegerHyperparameter("batch_size", 30, 300, default_value=200)
    learning_rate = CategoricalHyperparameter(
        "learning_rate",
        ["constant", "invscaling", "adaptive"],
        default_value="constant",
    )
    learning_rate_init = UniformFloatHyperparameter("learning_rate_init", 0.0001, 1.0, default_value=0.001, log=True)

    # Add all hyperparameters at once:
    cs.add_hyperparameters(
        [
            n_layer,
            n_neurons,
            activation,
            solver,
            batch_size,
            learning_rate,
            learning_rate_init,
        ]
    )

    # Adding conditions to restrict the hyperparameter space
    # Since learning rate is used when solver is 'sgd'
    use_lr = CS.conditions.EqualsCondition(child=learning_rate, parent=solver, value="sgd")
    # Since learning rate initialization will only be accounted for when using 'sgd' or 'adam'
    use_lr_init = CS.conditions.InCondition(child=learning_rate_init, parent=solver, values=["sgd", "adam"])
    # Since batch size will not be considered when optimizer is 'lbfgs'
    use_batch_size = CS.conditions.InCondition(child=batch_size, parent=solver, values=["sgd", "adam"])

    # We can also add  multiple conditions on hyperparameters at once:
    cs.add_conditions([use_lr, use_batch_size, use_lr_init])

    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "wallclock-limit": 100,  # max duration to run the optimization (in seconds)
            "cs": cs,  # configuration space
            "deterministic": True,
            # Uses pynisher to limit memory and runtime
            # Alternatively, you can also disable this.
            # Then you should handle runtime and memory yourself in the TA
            "limit_resources": False,
            "cutoff": 30,  # runtime limit for target algorithm
            "memory_limit": 3072,  # adapt this to reasonable value for your hardware
        }
    )

    # Max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_epochs = 50

    # Intensifier parameters
    intensifier_kwargs = {"initial_budget": 5, "max_budget": max_epochs, "eta": 3}

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=mlp_from_cfg,
        intensifier_kwargs=intensifier_kwargs,
    )

    tae = smac.get_tae_runner()

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = tae.run(config=cs.get_default_configuration(), budget=max_epochs, seed=0)[1]

    print("Value for default configuration: %.4f" % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = tae.run(config=incumbent, budget=max_epochs, seed=0)[1]

    print("Optimized Value: %.4f" % inc_value)
