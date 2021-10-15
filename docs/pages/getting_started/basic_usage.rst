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
:ref:`Branin <branin-example>`-example to see how to use it.

SMAC is called via the commandline with the following arguments:

.. code-block:: bash

    python3 smac.py --scenario SCENARIO --seed INT --verbose_level LEVEL --mode MODE

Required:
    :scenario: 
        Path to the file that specifies the scenario for this SMAC run.
Optional:
    :seed:
        The integer that the random-generator will be based upon. **Default**: 12345.

    :verbose_level:
        In [INFO, DEBUG]. Specifies the logging-verbosity. **Default**: INFO.

    :mode:
        In [SMAC, ROAR]. SMAC will use the bayeasian optimization with an intensification process,
        whereas ROAR stands for Random Online Adaptive Racing. **Default**: SMAC3.
        
    :restore_state:
        A string specifying the folder of the *SMAC*-run to be continued. Assuming exactly the same scenario, except for budget-options.

In the scenario file, there are two mandatory parameters: The ``algo`` parameter
defines how SMAC will call the target algorithm. Parameters will be appended to the call
with ``-PARAMETER VALUE``. So make sure your algorithm will accept the parameters in this
form. Read more in the section on :ref:`target algorithms <Target Algorithm Evaluator>`.
The ``paramfile`` parameter defines the path to the PCS-file <paramcs>,
which describes the ranges and default values of the tunable parameters.
Both will interpret paths from the execution-directory.


Currently, running SMAC via the commandline will register the algorithm with a :ref:`Target
Algorithm Evaluator<Target Algorithm Evaluator>`, that requires the target algorithm to print the
results to the console in the following format (see :ref:`Branin <branin-example>`):
    
.. code-block:: bash

    Result for SMAC: <STATUS>, <runtime>, <runlength>, <quality>, <seed>, <instance-specifics>
