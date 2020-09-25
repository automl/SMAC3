"""
================================================
Optimizing an MLP with Parallel SuccesiveHalving
================================================
An example for the usage of a model-free SuccessiveHalving intensifier in SMAC,
for parallel execution. The configurations are randomly sampled.

This examples uses a real-valued SuccessiveHalving through epochs.

4 workers are allocated for this run. As soon as any worker is idle,
SMAC internally creates more SuccessiveHalving instances to take
advantage of the idle resources.
"""

import logging

import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.configspace import ConfigurationSpace
from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario
from smac.intensification.successive_halving import SuccessiveHalving
from smac.initial_design.random_configuration_design import RandomConfigurations

# --------------------------------------------------------------
# We need to provide a pickable function and use __main__
# to be compliant with multiprocessing API
# Below is a work around to have a packaged function called
# mlp_from_cfg_func
# --------------------------------------------------------------
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from mlp_from_cfg_func import mlp_from_cfg  # noqa: E402
# --------------------------------------------------------------

if __name__ == '__main__':

    logger = logging.getLogger("MLP-example")
    logging.basicConfig(level=logging.INFO)

    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()

    # We can add multiple hyperparameters at once:
    n_layer = UniformIntegerHyperparameter("n_layer", 1, 4, default_value=1)
    n_neurons = UniformIntegerHyperparameter("n_neurons", 8, 512, log=True, default_value=10)
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
                         "cutoff": 20,  # runtime limit for target algorithm
                         "memory_limit": 3072,  # adapt this to reasonable value for your hardware
                         })

    # Intensification parameters
    # Intensifier will allocate from 5 to a maximum of 25 epochs to each configuration
    # Successive Halving child-instances are created to prevent idle
    # workers.
    intensifier_kwargs = {'initial_budget': 5, 'max_budget': 25, 'eta': 3,
                          'min_chall': 1, 'instance_order': 'shuffle_once'}

    # To optimize, we pass the function to the SMAC-object
    smac = ROAR(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=mlp_from_cfg,
                intensifier=SuccessiveHalving,
                intensifier_kwargs=intensifier_kwargs,
                initial_design=RandomConfigurations,
                n_jobs=4)

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(),
                                          instance='1', budget=25, seed=0)[1]
    print("Value for default configuration: %.4f" % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',
                                          budget=25, seed=0)[1]
    print("Optimized Value: %.4f" % inc_value)
