Facades
--------------------
SMAC of course itself offers a lot of design choices, some of which are crucial to achieve peak performance.
Luckily, often it is sufficient to distinguish between a few problem classes.
To make the usage of SMAC as easy as possible, we provide several facades designed for these different use cases.
Here we give some general recommendations on when to use which facade.
These recommendations are based on our experience and technical limitations and is by far not intended to be complete:

.. csv-table::
    :header: "", "SMAC4BB", "SMAC4HPO", "SMAC4MF", "SMAC4AC"
    :widths: 15, 10, 10, 10, 10

    "# parameter", "low", "low/medium/high", "low/medium/high", "low/medium/high"
    "categorical parameter", "yes", "supported", "supported", "supported"
    "conditional parameter", "yes", "supported", "supported", "supported"
    "instances", "no", "None or CV-folds", "None or CV-folds", "yes"
    "stochasticity",  "no", "supported", "supported", "supported"
    "objective", "any (except runtime)", "e.g. validation loss ",  "e.g. validation loss ", "e.g. runtime or quality"
    "multi-fidelity", "no", "no", "yes", "no"
    "parallelism", "no", "no", "no", "no"
    "search strategy", ":term:`GP` or :term:`GP-MCMC`", ":term:`RF`", ":term:`RF`", ":term:`RF`, :term:`GP`, :term:`GP-MCMC` or Random"


Here we show the class inheritance of the different facades. The different facades provide a interface to and configure
SMAC4AC for pre-set problem types.

.. figure:: ../../images/smac_facades_all_classes.png
    :width: 700px
    :align: center
    :alt: Class diagram of the SMAC facades.
    :figclass: align-center

    Class inheritance of the SMAC facades.

