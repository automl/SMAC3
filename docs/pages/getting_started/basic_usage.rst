Basic Usage
===========

There are two ways to use SMAC, either within your Python-code or over the commandline.

Either way, you need to provide a :ref:`target algorithm<Target Algorithm Evaluator>` you want to
optimize and the `configuration space <https://automl.github.io/ConfigSpace/master/>`_, which
specifies the legal ranges and default values of the tunable parameters.
In addition, you can configure the optimization process with the :ref:`scenario<Scenario>` object.


Python
~~~~~~

The usage of SMAC from your Python-code is described in the :ref:`minimal example<Minimal Example>`.
Scenario and configuration space are both build within the code. The target algorithm needs to be
registered with a :ref:`target algorithm<Target Algorithm Evaluator>`, which communicates between
SMAC and the target algorithm. To optimize a function, you can instantiate ``ExecuteTAFuncDict`` or
``ExecuteTAFuncArray``. In both cases, the algorithm needs to return a cost, representing the
quality of the solution. Time- and memory-limits, on the other hand, are measured and enforced by
Pynisher.


Commandline
~~~~~~~~~~~

To use SMAC via the commandline, you need a :ref:`scenario-file <Scenario>` and a :term:`PCS-file <PCS>`.
The script to invoke SMAC is located in *scripts/smac*. Please see the
:ref:`Branin`-example to see how to use it.

SMAC is called via the commandline with the following arguments:

.. code-block:: bash

    python smac.py --scenario SCENARIO --seed INT --verbose_level LEVEL --mode MODE

Please refer to :ref:`arguments<Arguments>` for more options.

In the scenario file, there are two mandatory parameters: The ``algo`` parameter
defines how SMAC will call the target algorithm. Parameters will be appended to the call
with ``-PARAMETER VALUE``. So make sure your algorithm will accept the parameters in this
form. Read more in the section :ref:`here<Target Algorithm Evaluator>`.
The ``paramfile`` parameter defines the path to the :term:`PCS` file,
which describes the ranges and default values of the tunable parameters.
Both will interpret paths from the execution-directory.

Currently, running SMAC via the commandline will register the algorithm with a :ref:`Target
Algorithm Evaluator<Target Algorithm Evaluator>`, that requires the target algorithm to print the
results to the console in the following format (see :ref:`Branin`):
    
.. code-block:: bash

    Result for SMAC: <STATUS>, <runtime>, <runlength>, <quality>, <seed>, <instance-specifics>
