Manual
======
.. role:: bash(code)
    :language: bash

In the following we explain how to use **SMAC3**. We explain the framework and
the options you can use to configure *SMAC*.

*SMAC* stands for **Sequential Model-Based Algorithm Configuration** and uses
Bayesian optimization to configure the hyperparameters of an algorithm. Instance
features can be used to optimize the algorithm on a certain set of instances.

There are two ways to use *SMAC*:
You can use the :ref:`command line <commandline>` to optimize
algorithms that are invoked via a bash command,
but you can also use *SMAC* :ref:`directly in Python <inpython>`.

We provide examples for the usage of *SMAC* in the :ref:`Quickstart guide <branin-example>`.

*SMAC* is written in Python 3 and hosted on `GitHub <https://github.com/automl/SMAC3/>`_.

.. toctree::

    basic_usage
    options
    tae
    instances
    psmac
    validation
