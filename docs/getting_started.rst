Getting Started
===============

In the core, SMAC needs four components to run an optimization process, which are explained on this page. The following
four components are:

* Configuration space
* Target algorithm 
* Scenario
* Facade


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


Target Algorithm
----------------

The target algorithm takes a configuration from the configuration space and returns a performance value.
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

    In general, the arguments of the target algorithm depend on the intensifier. However,
    in all cases, the first argument must be the configuration (arbitrary argument name is possible here) and a seed.
    If you specified instances in the scenario, SMAC requires `instance` as argument additionally. If you use
    `SuccessiveHalving` or `Hyperband` as intensifier but you did not specify instances, SMAC passes `budget` as
    argument to the target function.


.. warning::

    SMAC passes either `instance` or `budget` to the target function but never both.


Scenario
--------

The :ref:`Scenario<smac.scenario.Scenario>` is used to provide environment variables. For example, 
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

:ref:`Facades<smac.facade>` are used to satisfy different tasks. We support the following facades by default:

* :ref:`BlackBoxFacade<smac.facade.blackbox_facade.BlackBoxFacade>`: Random Search
* :ref:`HyperbandFacade<smac.facade.hyperband_facade.HyperbandFacade>`: Random Search when using multiple budgets (like epochs or subset sizes).
* :ref:`BlackBoxFacade<smac.facade.blackbox_facade.BlackBoxFacade>`: Black-box optimization
* :ref:`HyperparameterFacade<smac.facade.hyperparameter_facade.HyperparameterFacade>`: Hyperparameter optimization
* :ref:`MultiFidelityFacade<smac.facade.multi_fidelity_facade.MultiFidelityFacade>`: Multi-Fidelity optimization when using multiple budgets (like epochs or subset sizes).
* :ref:`AlgorithmConfigurationFacade<smac.facade.algorithm_configuration_facade.AlgorithmConfigurationFacade>`: Algorithm Configuration to optimize across different instances.

Each facade might differ in the following ways (but are not limited to that):
- Initial design (how the first configurations are sampled)
- Surrogate model (which model is used to learn promising regions on the hyperparamter facade)
- Acqusition function (exploration and exploitation trade-off)

.. code-block:: python

    from smac import BlackBoxFacade

    smac = BlackBoxFacade(
        scenario=scenario,
        target_algorithm=train,
        ...
    )


.. note::

    We want to emphasize that SMAC is a highly modular optimization framework.
    The facade accepts many arguments to specify components of the pipeline.
