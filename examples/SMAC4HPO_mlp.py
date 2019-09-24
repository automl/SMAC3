"""
An example for the usage of SMAC within Python.
We optimize a simple MLP on the MNIST digits dataset using "Hyperband" intensification.
"""

import logging
import warnings

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning

import ConfigSpace as CS
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.intensification.hyperband import Hyperband

digits = load_digits()


# Target Algorithm
def mlp_from_cfg(cfg, seed, instance, budget, **kwargs):
    """
        Creates a MLP classifier from sklearn and fits the given data on it.
        This is the function-call we try to optimize. Chosen values are stored in
        the configuration (cfg).

        Parameters
        ----------
        cfg: Configuration
            configuration chosen by smac
        seed: int or RandomState
            used to initialize the rf's random generator
        instance: str
            used to represent the instance to use (just a placeholder for this example)
        budget: float
            used to set max iterations for the MLP

        Returns
        -------
        float
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the MLP, so we replace them with placeholder values.
    lr = cfg['learning_rate'] if cfg['learning_rate'] else 'constant'
    lr_init = cfg['learning_rate_init'] if cfg['learning_rate_init'] else 0.001
    batch_size = cfg['batch_size'] if cfg['batch_size'] else 200

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        mlp = MLPClassifier(
            hidden_layer_sizes=[cfg["n_neurons"]] * cfg["n_layer"],
            solver=cfg['solver'],
            batch_size=batch_size,
            activation=cfg['activation'],
            learning_rate=lr,
            learning_rate_init=lr_init,
            max_iter=int(np.ceil(budget)),
            random_state=seed)

        # returns the cross validation accuracy
        score = cross_val_score(mlp, digits.data, digits.target, cv=5)

    return 1 - np.mean(score)  # Because minimize!


logger = logging.getLogger("MLP-example")
logging.basicConfig(level=logging.INFO)
logger.info("Running MLP example for SMAC. If you experience "
            "difficulties, try to decrease the memory-limit.")

# Build Configuration Space which defines all parameters and their ranges.
# To illustrate different parameter types,
# we use continuous, integer and categorical parameters.
cs = ConfigurationSpace()

# We can add multiple hyperparameters at once:
n_layer = UniformIntegerHyperparameter("n_layer", 1, 5, default_value=2)
n_neurons = UniformIntegerHyperparameter("n_neurons", 8, 1024, log=True, default_value=10)
activation = CategoricalHyperparameter("activation", ['logistic', 'tanh', 'relu'],
                                       default_value='relu')
solver = CategoricalHyperparameter('solver', ['lbfgs', 'sgd', 'adam'], default_value='adam')
batch_size = UniformIntegerHyperparameter('batch_size', 30, 300, default_value=200)
learning_rate = CategoricalHyperparameter('learning_rate', ['constant', 'invscaling', 'adaptive'],
                                          default_value='constant')
learning_rate_init = UniformFloatHyperparameter('learning_rate_init', 0.0001, 1.0, default_value=0.001, log=True)
cs.add_hyperparameters([n_layer, n_neurons, activation, solver, batch_size, learning_rate, learning_rate_init])

# Adding conditions to restrict the hyperparameter space
# Since learning rate is used when solver is 'sgd'
use_lr = CS.conditions.EqualsCondition(child=learning_rate, parent=solver, value='sgd')
# Since learning rate initialization will only be accounted for when using 'sgd' or 'adam'
use_lr_init = CS.conditions.InCondition(child=learning_rate_init, parent=solver, values=['sgd', 'adam'])
# Since batch size will not be considered when optimizer is 'lbfgs'
use_batch_size = CS.conditions.InCondition(child=batch_size, parent=solver, values=['sgd', 'adam'])
# We can also add  multiple conditions on hyperparameters at once:
cs.add_conditions([use_lr, use_batch_size, use_lr_init])

# setting cutoff
# normally, it is a runtime cutoff, but in this case, we use cutoff to represent the number of epochs for MLP
cutoff = 50

# SMAC scenario object
scenario = Scenario({"run_obj": "quality",     # we optimize quality (alternative runtime)
                     "wallclock-limit": 100,   # max duration to run the optimization (in seconds)
                     "cs": cs,                 # configuration space
                     "deterministic": "true",
                     "limit_resources": False,  # Disables pynisher to pass cutoff directly to the target algorithm.
                                                # Timeouts have to be taken care within the TA
                     })

# intensifier parameters
intensifier_kwargs = {'initial_budget': 5, 'max_budget': cutoff, 'eta': 3}
# To optimize, we pass the function to the SMAC-object
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=mlp_from_cfg,
                intensifier=Hyperband,                  # you can also change the intensifier to use like this!
                                                        # This example currently uses Hyperband intensification,
                intensifier_kwargs=intensifier_kwargs)  # all parameters related to intensifier can be passed like this

# Example call of the function with default values
# It returns: Status, Cost, Runtime, Additional Infos
def_value = smac.get_tae_runner().run(cs.get_default_configuration(), '1', cutoff, 0)[1]
print("Value for default configuration: %.4f" % (-1*def_value))

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = smac.get_tae_runner().run(incumbent, '1', cutoff, 0)[1]
print("Optimized Value: %.4f" % (-1*inc_value))
