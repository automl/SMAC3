Getting Started
===============

In the core, SMAC needs four components to run an optimization process, all of which are explained on this page.


Configuration Space
-------------------

The configuration space defines the search space of the hyperparameters and, therefore, the tunable parameters' legal
ranges and default values.

.. code-block:: python
    
    from ConfigSpace import ConfigSpace

    cs = ConfigSpace({
        "learning_rate" (1e-3, 1e-1),
    })

Please see the documentation of ``ConfigSpace`` for more details.


Target Function
---------------

The target function takes a configuration from the configuration space and returns a performance value.
For example, you could use a Neural Network and predict the performance based on the learning rate. Every configuration
would (most likely) return a different value. However, SMAC tries to find the best learning rate by trying 
different and potentially improving configurations.

.. code-block:: python
    
    def train(self, config: Configuration, seed: int) -> float:
        model = MultiLayerPerceptron(learning_rate=config["learning_rate"])
        model.fit(...)
        accuracy = model.validate(...)

        return 1 - accuracy  # SMAC always minimizes (the smaller the better)

.. note::

    In general, the arguments of the target function depend on the intensifier. However,
    in all cases, the first argument must be the configuration (arbitrary argument name is possible here) and a seed.
    If you specified instances in the scenario, SMAC requires ``instance`` as argument additionally. If you use
    ``SuccessiveHalving`` or ``Hyperband`` as intensifier but you did not specify instances, SMAC passes `budget` as
    argument to the target function.


.. warning::

    SMAC passes either `instance` or `budget` to the target function but never both.


Scenario
--------

The :ref:`Scenario<smac.scenario>` is used to provide environment variables. For example, 
you want to limit the optimization process by a time limit or want to specify where to save the results. 

.. code-block:: python

    from smac import Scenario

    scenario = Scenario(
        configspace=cs,
        output_directory=Path("output_directory")
        walltime_limit=120,  # Limit to two minutes
        n_workers=8,  # Use eight workers
        ...
    )


Facade
------

A :ref:`facade<smac.facade>` is the entry point to SMAC, which constructs a default optimization 
pipeline for you. SMAC offers various facades, which satisfy many use cases and are crucial to 
achieving peak performance. The idea behind the facades is to provide a simple interface to SMAC, 
which is easy to use and understand without diving deep into the material. However, experts are 
invited to change the components as they please to achieve even better performance potentially. The following 
table (horizontal scrollable) shows you what is supported and reveals the default :ref:`components<Components>`:

.. csv-table::
    :header: "", ":ref:`Black-Box<smac.facade.blackbox_facade>`", ":ref:`Hyperparameter Optimization<smac.facade.hyperparameter_optimization_facade>`", ":ref:`Multi-Fidelity<smac.facade.multi_fidelity_facade>`", ":ref:`Algorithm Configuration<smac.facade.algorithm_configuration_facade>`", ":ref:`Random<smac.facade.random_facade>`", ":ref:`Hyperband<smac.facade.hyperband_facade>`"

    "#Parameters", "low", "low/medium/high", "low/medium/high", "low/medium/high", "low/medium/high", "low/medium/high"
    "Supports Instances", "❌", "✅", "✅", "✅", "❌", "✅"
    "Supports Multi-Fidelity", "❌", "❌", "✅", "✅", "❌", "✅"
    "Initial Design", ":ref:`Sobol<smac.initial_design.sobol_design>`", ":ref:`Sobol<smac.initial_design.sobol_design>`", ":ref:`Random<smac.initial_design.random_design>`", ":ref:`Default<smac.initial_design.default_design>`", ":ref:`Default<smac.initial_design.default_design>`", ":ref:`Default<smac.initial_design.default_design>`"
    "Surrogate Model", ":ref:`Gaussian Process<smac.model.gaussian_process.gaussian_process>`", ":ref:`Random Forest<smac.model.random_forest.random_forest>`", ":ref:`Random Forest<smac.model.random_forest.random_forest>`", ":ref:`Random Forest<smac.model.random_forest.random_forest>`", "Not used", "Not used"
    "Acquisition Function", ":ref:`Expected Improvement<smac.acquisition.function.expected_improvement>`", ":ref:`Expected Improvement<smac.acquisition.function.expected_improvement>`", ":ref:`Expected Improvement<smac.acquisition.function.expected_improvement>`", ":ref:`Expected Improvement<smac.acquisition.function.expected_improvement>`", "Not used", "Not used"
    "Acquisition Maximier", ":ref:`Local and Random Search<smac.acquisition.maximizers.local_and_random_search>`", ":ref:`Local and Random Search<smac.acquisition.maximizers.local_and_random_search>`", ":ref:`Local and Random Search<smac.acquisition.maximizers.local_and_random_search>`", ":ref:`Local and Random Search<smac.acquisition.maximizers.local_and_random_search>`", ":ref:`Local and Random Search<smac.acquisition.maximizers.random_search>`", ":ref:`Local and Random Search<smac.acquisition.maximizers.random_search>`"
    "Intensifier", ":ref:`Default<smac.intensifier.intensifier>`", ":ref:`Default<smac.intensifier.intensifier>`", ":ref:`Hyperband<smac.intensifier.hyperband>`", ":ref:`Hyperband<smac.intensifier.hyperband>`", ":ref:`Default<smac.intensifier.intensifier>`", ":ref:`Hyperband<smac.intensifier.hyperband>`",
    "Runhistory Encoder", ":ref:`Default<smac.runhistory.encoder.encoder>`", ":ref:`Log<smac.runhistory.encoder.log_encoder>`", ":ref:`Log<smac.runhistory.encoder.log_encoder>`", ":ref:`Default<smac.runhistory.encoder.encoder>`", ":ref:`Default<smac.runhistory.encoder.encoder>`", ":ref:`Default<smac.runhistory.encoder.encoder>`"
    "Random Design Probability", "8.5%", "20%", "20%", "50%", "Not used", "Not used"


.. note::

    The multi-fidelity facade is the closest implementation to `BOHB <https://github.com/automl/HpBandSter>`_.


.. note::

    We want to emphasize that SMAC is a highly modular optimization framework.
    The facade accepts many arguments to specify components of the pipeline.


The facades can be imported directely from the ``smac`` module.

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


