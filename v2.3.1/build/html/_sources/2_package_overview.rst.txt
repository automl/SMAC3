Package Overview 
================

SMAC supports you in determining well-performing hyperparameter configurations for your algorithms. By being a robust 
and flexible framework for :term:`Bayesian Optimization<BO>`, SMAC can improve performance within a few function
evaluations. It offers several entry points and pre-sets for typical use cases, such as optimizing
hyperparameters, solving low dimensional continuous (artificial) global optimization problems and configuring algorithms 
to perform well across multiple problem :term:`instances<Instances>`.


Features
--------

SMAC has the following characteristics and capabilities:

Global Optimizer
    :term:`Bayesian Optimization` is used for sample-efficient optimization.

Optimize :term:`Black-Box` Functions
    Optimization is only aware of input and output. It is agnostic to internals of the function.

Flexible Hyperparameters
    Use categorical, continuous, hierarchical and/or conditional hyperparameters with the well-integrated
    `ConfigurationSpace <https://automl.github.io/ConfigSpace>`_. SMAC can optimize *up to 100 hyperparameters*
    efficiently.

Any Objectives
    Optimization with any :term:`objective<Objective>` (e.g., accuracy, runtime, cross-validation, ...) is possible.

:ref:`Multi-Objective<Multi-Objective Optimization>`
    Optimize arbitrary number of objectives using scalarized multi-objective algorithms. Both ParEGO [Know06]_ and
    mean aggregation strategies are supported.

:ref:`Multi-Fidelity<Multi-Fidelity Optimization>` Optimization
    Judge configurations on multiple :term:`budgets<Budget>` to discard unsuitable configurations
    early on. This will result in a massive speed-up, depending on the budgets.
    
:ref:`Instances<Optimization across Instances>`
    Find well-performing hyperparameter configurations not only for one instance (e.g. dataset) of
    an algorithm, but for many.
    
Command-Line Interface
    SMAC can not only be executed within a python file but also from the commandline. Consequently,
    not only algorithms in python can be optimized, but implementations in other languages as well.

    .. note ::

        Command-line interface has been temporarily disabled in v2.0. Please fall back to v1.4 if you need it.


Comparison
----------

The following table provides an overview of SMAC's capabilities in comparison with other optimization tools.

.. csv-table::
    :header: "Package", "Complex Hyperparameter Space", ":term:`Multi-Objective` ", ":term:`Multi-Fidelity`", ":term:`Instances`", "Command-Line Interface", "Parallelism"
    :widths: 10, 10, 10, 10, 10, 10, 10

    HyperMapper, ✅, ✅, ❌, ❌, ❌, ❌
    Optuna, ✅, ✅, ✅, ❌, ✅, ✅
    Hyperopt, ✅, ❌, ❌, ❌, ✅, ✅
    BoTorch, ❌, ✅, ✅, ❌, ❌, ✅
    OpenBox, ✅, ✅, ❌, ❌, ❌, ✅
    HpBandSter, ✅, ❌, ✅, ❌, ❌, ✅
    SMAC, ✅, ✅, ✅, ✅, ✅, ✅
