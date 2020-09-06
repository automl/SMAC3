"""
================================
Optimizing an MLP with Hyperband
================================

An example for the usage of a model-free Hyperband intensifier in SMAC.
The configurations are randomly sampled

In this example, we use a real-valued budget in hyperband (number of epochs to train the MLP) and
optimize the average accuracy on a 5-fold cross validation.
"""

import logging

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

import numpy as np

from smac.configspace import ConfigurationSpace
from smac.facade.hyperband_facade import HB4AC
from smac.scenario.scenario import Scenario
# --------------------------------------------------------------
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from mlp_from_cfg_func import mlp_from_cfg  # noqa: E402
# --------------------------------------------------------------


logger = logging.getLogger("MLP-example")
logging.basicConfig(level=logging.INFO)

# Build Configuration Space which defines all parameters and their ranges.
# To illustrate different parameter types,
# we use continuous, integer and categorical parameters.
cs = ConfigurationSpace()

# We can add multiple hyperparameters at once:
n_layer = UniformIntegerHyperparameter("n_layer", 1, 5, default_value=1)
n_neurons = UniformIntegerHyperparameter("n_neurons", 8, 1024, log=True, default_value=10)
activation = CategoricalHyperparameter("activation", ['logistic', 'tanh', 'relu'],
                                       default_value='tanh')
batch_size = UniformIntegerHyperparameter('batch_size', 30, 300, default_value=200)
learning_rate_init = UniformFloatHyperparameter('learning_rate_init', 0.0001, 1.0, default_value=0.001, log=True)
cs.add_hyperparameters([n_layer, n_neurons, activation, batch_size, learning_rate_init])

# SMAC scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                     "wallclock-limit": 100,  # max duration to run the optimization (in seconds)
                     "cs": cs,  # configuration space
                     "deterministic": "true",
                     "limit_resources": True,  # Uses pynisher to limit memory and runtime
                     # Alternatively, you can also disable this.
                     # Then you should handle runtime and memory yourself in the TA
                     "cutoff": 30,  # runtime limit for target algorithm
                     "memory_limit": 3072,  # adapt this to reasonable value for your hardware
                     })

# max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
max_iters = 50
# intensifier parameters
intensifier_kwargs = {'initial_budget': 5, 'max_budget': max_iters, 'eta': 3}
# To optimize, we pass the function to the SMAC-object
smac = HB4AC(scenario=scenario, rng=np.random.RandomState(42),
             tae_runner=mlp_from_cfg,
             intensifier_kwargs=intensifier_kwargs)  # all arguments related to intensifier can be passed like this

# Example call of the function with default values
# It returns: Status, Cost, Runtime, Additional Infos
def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(),
                                      instance='1', budget=max_iters, seed=0)[1]
print("Value for default configuration: %.4f" % def_value)

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',
                                      budget=max_iters, seed=0)[1]
print("Optimized Value: %.4f" % inc_value)
