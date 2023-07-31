Getting Started
===============

SMAC needs four core components (configuration space, target function, scenario and a facade) to run an
optimization process, all of which are explained on this page.

They interact in the following way:

.. image:: images/smac_components_interaction.jpg
  :width: 400
  :alt: Interaction of SMAC's components


Configuration Space
-------------------

The configuration space defines the search space of the hyperparameters and, therefore, the tunable parameters' legal
ranges and default values.

.. code-block:: python
    
    from ConfigSpace import ConfigSpace

    cs = ConfigurationSpace({
        "myfloat": (0.1, 1.5),                # Uniform Float
        "myint": (2, 10),                     # Uniform Integer
        "species": ["mouse", "cat", "dog"],   # Categorical
    })

Please see the documentation of `ConfigSpace <https://automl.github.io/ConfigSpace/main/>`_ for more details.


Target Function
---------------

The target function takes a configuration from the configuration space and returns a performance value.
For example, you could use a Neural Network to predict on your data and get some validation performance.
If, for instance, you would tune the learning rate of the Network's optimizer, every learning rate will
change the final validation performance of the network. This is the target function.
SMAC tries to find the best performing learning rate by trying different values and evaluating the target function -
in an efficient way.

.. code-block:: python
    
    def train(self, config: Configuration, seed: int) -> float:
        model = MultiLayerPerceptron(learning_rate=config["learning_rate"])
        model.fit(...)
        accuracy = model.validate(...)

        return 1 - accuracy  # SMAC always minimizes (the smaller the better)

.. warning::

    SMAC *always* minimizes the value returned from the target function.


.. note::

    In general, the arguments of the target function depend on the intensifier. However,
    in all cases, the first argument must be the configuration (arbitrary argument name is possible here) and a seed.
    If you specified instances in the scenario, SMAC requires ``instance`` as argument additionally. If you use
    ``SuccessiveHalving`` or ``Hyperband`` as intensifier but you did not specify instances, SMAC passes `budget` as
    argument to the target function. But don't worry: SMAC will tell you if something is missing or if something is not
    used.


.. warning::

    SMAC passes either `instance` or `budget` to the target function but never both.


Scenario
--------

The :ref:`Scenario<smac.scenario>` is used to provide environment variables. For example, 
if you want to limit the optimization process by a time limit or want to specify where to save the results.

.. code-block:: python

    from smac import Scenario

    scenario = Scenario(
        configspace=cs,
        output_directory=Path("your_output_directory")
        walltime_limit=120,  # Limit to two minutes
        n_trials=500,  # Evaluated max 500 trials
        n_workers=8,  # Use eight workers
        ...
    )


Facade
------

A :ref:`facade<smac.facade>` is the entry point to SMAC, which constructs a default optimization 
pipeline for you. SMAC offers various facades, which satisfy many common use cases and are crucial to
achieving peak performance. The idea behind the facades is to provide a simple interface to all of SMAC's components,
which is easy to use and understand and without the need of deep diving into the material. However, experts are
invited to change the components to their specific hyperparameter optimization needs. The following
table (horizontally scrollable) shows you what is supported and reveals the default :ref:`components<Components>`:


.. csv-table::
    :header: "", ":ref:`Black-Box<smac.facade.blackbox\\_facade>`", ":ref:`Hyperparameter Optimization<smac.facade.hyperparameter\\_optimization\\_facade>`", ":ref:`Multi-Fidelity<smac.facade.multi\\_fidelity\\_facade>`", ":ref:`Algorithm Configuration<smac.facade.algorithm\\_configuration\\_facade>`", ":ref:`Random<smac.facade.random\\_facade>`", ":ref:`Hyperband<smac.facade.hyperband\\_facade>`"

    "#Parameters", "low", "low/medium/high", "low/medium/high", "low/medium/high", "low/medium/high", "low/medium/high"
    "Supports Instances", "❌", "✅", "✅", "✅", "❌", "✅"
    "Supports Multi-Fidelity", "❌", "❌", "✅", "✅", "❌", "✅"
    "Initial Design", ":ref:`Sobol<smac.initial\\_design.sobol\\_design>`", ":ref:`Sobol<smac.initial\\_design.sobol\\_design>`", ":ref:`Random<smac.initial\\_design.random\\_design>`", ":ref:`Default<smac.initial\\_design.default\\_design>`", ":ref:`Default<smac.initial\\_design.default\\_design>`", ":ref:`Default<smac.initial\\_design.default\\_design>`"
    "Surrogate Model", ":ref:`Gaussian Process<smac.model.gaussian\\_process.gaussian\\_process>`", ":ref:`Random Forest<smac.model.random\\_forest.random\\_forest>`", ":ref:`Random Forest<smac.model.random\\_forest.random\\_forest>`", ":ref:`Random Forest<smac.model.random\\_forest.random\\_forest>`", "Not used", "Not used"
    "Acquisition Function", ":ref:`Expected Improvement<smac.acquisition.function.expected\\_improvement>`", ":ref:`Log Expected Improvement<smac.acquisition.function.expected\\_improvement>`", ":ref:`Log Expected Improvement<smac.acquisition.function.expected\\_improvement>`", ":ref:`Expected Improvement<smac.acquisition.function.expected\\_improvement>`", "Not used", "Not used"
    "Acquisition Maximizer", ":ref:`Local and Sorted Random Search<smac.acquisition.maximizer.local\\_and\\_random\\_search>`", ":ref:`Local and Sorted Random Search<smac.acquisition.maximizer.local\\_and\\_random\\_search>`", ":ref:`Local and Sorted Random Search<smac.acquisition.maximizer.local\\_and\\_random\\_search>`", ":ref:`Local and Sorted Random Search<smac.acquisition.maximizer.local\\_and\\_random\\_search>`", ":ref:`Local and Sorted Random Search<smac.acquisition.maximizer.random\\_search>`", ":ref:`Local and Sorted Random Search<smac.acquisition.maximizer.random\\_search>`"
    "Intensifier", ":ref:`Default<smac.intensifier.intensifier>`", ":ref:`Default<smac.intensifier.intensifier>`", ":ref:`Hyperband<smac.intensifier.hyperband>`", ":ref:`Hyperband<smac.intensifier.hyperband>`", ":ref:`Default<smac.intensifier.intensifier>`", ":ref:`Hyperband<smac.intensifier.hyperband>`",
    "Runhistory Encoder", ":ref:`Default<smac.runhistory.encoder.encoder>`", ":ref:`Log<smac.runhistory.encoder.log\\_encoder>`", ":ref:`Log<smac.runhistory.encoder.log\\_encoder>`", ":ref:`Default<smac.runhistory.encoder.encoder>`", ":ref:`Default<smac.runhistory.encoder.encoder>`", ":ref:`Default<smac.runhistory.encoder.encoder>`"
    "Random Design Probability", "8.5%", "20%", "20%", "50%", "Not used", "Not used"


.. note::

    The multi-fidelity facade is the closest implementation to `BOHB <https://github.com/automl/HpBandSter>`_.


.. note::

    We want to emphasize that SMAC is a highly modular optimization framework.
    The facade accepts many arguments to specify components of the pipeline. Please also note, that in contrast
    to previous versions, instantiated objects are passed instead of *kwargs*.


The facades can be imported directly from the ``smac`` module.

.. code-block:: python

    from smac import BlackBoxFacade as BBFacade
    from smac import HyperparameterOptimizationFacade as HPOFacade
    from smac import MultiFidelityFacade as MFFacade
    from smac import AlgorithmConfigurationFacade as ACFacade
    from smac import RandomFacade as RFacade
    from smac import HyperbandFacade as HBFacade

    smac = HPOFacade(scenario=scenario, target_function=train)
    smac = MFFacade(scenario=scenario, target_function=train)
    smac = ACFacade(scenario=scenario, target_function=train)
    smac = RFacade(scenario=scenario, target_function=train)
    smac = HBFacade(scenario=scenario, target_function=train)
