Facades
-------

SMAC of course itself offers a lot of design choices, some of which are crucial to achieve peak performance.
Luckily, often it is sufficient to distinguish between a few problem classes.
To make the usage of SMAC as easy as possible, we provide several facades designed for these different use cases.
Here we give some general recommendations on when to use which facade.
These recommendations are based on our experience and technical limitations and is by far not intended to be complete:

.. csv-table::
    :header: "", "SMAC4BB", "SMAC4HPO", "SMAC4MF", "SMAC4AC"
    :widths: 15, 10, 10, 10, 10

    "# parameter", "low", "low/medium/high", "low/medium/high", "low/medium/high"
    "Categorical parameters", "yes", "supported", "supported", "supported"
    "Conditional parameters", "yes", "supported", "supported", "supported"
    "Instances", "no", "None or CV-folds", "None or CV-folds", "yes"
    "Stochasticity",  "no", "supported", "supported", "supported"
    "Objective", "any (except runtime)", "e.g. validation loss ",  "e.g. validation loss ", "any"
    "Multi-Fidelity", "no", "no", "yes", "yes"
    "Search Strategy", ":term:`Gaussian Process<GP>` or :term:`GP-MCMC`", ":term:`Random Forest<RF>`", ":term:`Random
    Forest<RF>`", ":term:`Random Forest<RF>`, :term:`Gaussian Process<GP>`, :term:`GP-MCMC` or Random"


Inheritance
~~~~~~~~~~~

Here we show the class inheritance of the different facades.
Because SMAC4AC is the facade every other facade is inherited from, we recommend using SMAC4AC if a lot of flexibility is needed. 

.. figure:: ../../images/smac_facades_all_classes.png
    :width: 100%
    :align: center
    :alt: Class diagram of the SMAC facades.
    :figclass: align-center

    Class inheritance of the SMAC facades.

