Command-Line Interface
======================

The command-line interface enables the user to run target functions which are non-python code. 
The passed and further called script (using `Popen`) needs to return a standard output which is then interpreted 
to perform the optimization process. 

.. note ::

    In SMAC v2.0, SMAC can not be called from the command-line directly. Instead, the user should use the python 
    interface to call SMAC. The command-line interface is still available in SMAC v1.4.



Call of the Target Function
---------------------------

The following example shows how the script is called:

.. code-block:: bash

    filename --instance=test --instance_features=test --seed=0 --hyperparameter1=5323

However, as for the python target function, the arguments like instance or budget are depending on which
components are used. The hyperparameters are depending on the configuration space. The variable ``filename`` could be 
something like ``./path/to/your/script.sh``.

We recommend using the following code to receive the arguments in a bash script. Please note that the user is not limited
to bash scripts but can also use executables, python scripts or anything else.

.. note ::

    Since the script is called wih the filename only, make sure to mark the type of the file (e.g., ``#!/bin/bash`` 
    or ``#!/usr/bin/env python``).

.. warning ::

    Everytime an instance is passed, also an instance feature in form of a comma-separated list (no spaces) of floats is
    passed. If no instance feature for the instance is given, an empty list is passed.


.. code-block:: bash

    #!/bin/bash

    # Set arguments first
    for argument in "$@"
    do
        key=$(echo $argument | cut -f1 -d=)
        value=$(echo $argument | cut -f2 -d=)   

        if [[ $key == *"--"* ]]; then
            v="${key/--/}"
            declare $v="${value}" 
        fi
    done

    echo $instance
    echo $hyperparameter1


Return of the Target Function
-----------------------------

The script must return an stdout (echo or print) in the following form (white-spaces are ignored):

.. code-block:: 

    cost=0.5; runtime=0.01; status=SUCCESS; additional_info=test (single-objective)
    cost=0.5, 0.4; runtime=0.01; status=SUCCESS; additional_info=test (multi-objective)

All arguments are optional except cost and are separated by a semicolon. The string of the status must match
one of the values from :ref:`StatusType<smac.runhistory.enumerations>`.


Start the Optimization
----------------------

The optimization will be started by the normal python interface. The only difference is that you need to pass
a string as target function instead of a python function.

.. warning ::

    Your script needs to have rights to be executed (e.g., update the rights with ``chmod``).

.. code-block:: python

    ...
    smac = BlackBoxFacade(scenario, target_function="./path/to/your/script.sh")
    incumbent = smac.optimize()
    ...

