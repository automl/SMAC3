Usage Recommendation
--------------------
SMAC of course itself offers a lot of design choices, some of which are crucial to achieve peak performance.
Luckily, often it is sufficient to distinguish between a few problem classes.
To make the usage of SMAC as easy as possible, we provide several facades designed for these different use cases.
Here we give some general recommendations on when to use which facade.
These recommendations are based on our experience and technical limitations and is by far not intended to be complete:

.. csv-table::
    :header: "Characteristic", "SMAC4BB", "SMAC4HPO", "SMAC4AC"
    :widths: 15, 10, 10, 10

    "# parameter", "low", "low/medium/high", "low/medium/high"
    "categorical parameter", "yes", "supported", "supported"
    "conditional parameter", "yes", "supported", "supported"
    "instances", "no", "None or CV-folds", "yes"
    "stochasticity",  "no", "supported", "supported"
    "objective", "any (except runtime)", "e.g. validation loss ", "e.g. runtime or quality"