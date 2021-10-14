Usage Recommendation
--------------------
SMAC of course itself offers a lot of design choices, some of which are crucial to achieve peak performance.
Luckily, often it is sufficient to distinguish between a few problem classes.
To make the usage of SMAC as easy as possible, we provide several facades designed for these different use cases.
Here we give some general recommendations on when to use which facade.
These recommendations are based on our experience and technical limitations and is by far not intended to be complete:

.. csv-table::
    :header: "Characteristic", "SMAC4BB", "SMAC4HPO", "SMAC4AC", "SMAC4MF"
    :widths: 15, 10, 10, 10, 10

    "# parameter", "low", "low/medium/high", "low/medium/high", "low/medium/high"
    "categorical parameter", "yes", "supported", "supported", "supported"
    "conditional parameter", "yes", "supported", "supported", "supported"
    "instances", "no", "None or CV-folds", "yes", "None or CV-folds"
    "stochasticity",  "no", "supported", "supported", "supported"
    "objective", "any (except runtime)", "e.g. validation loss ", "e.g. runtime or quality", "e.g. validation loss "
    "multi-fidelity", "no", "no", "no", "yes"

What does None/Cv folds mean for instances??