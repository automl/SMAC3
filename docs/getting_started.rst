Getting Started
===============

There are two ways to use SMAC, either within your Python-code or over the commandline.

Either way, you need to provide a :ref:`target algorithm<Target Algorithm Evaluator>` you want to
optimize and the `configuration space <https://automl.github.io/ConfigSpace/master/>`_, which
specifies the legal ranges and default values of the tunable parameters.
In addition, you can configure the optimization process with the :ref:`scenario<Scenario>` object.

The usage of SMAC from your Python-code is described in the :ref:`minimal example<Minimal Example>`.
Scenario and configuration space are both build within the code. The target algorithm needs to be
registered with a :ref:`target algorithm<Target Algorithm Evaluator>`, which communicates between
SMAC and the target algorithm. To optimize a function, you can instantiate ``ExecuteTAFuncDict`` or
``ExecuteTAFuncArray``. In both cases, the algorithm needs to return a cost, representing the
quality of the solution. Time- and memory-limits, on the other hand, are measured and enforced by
Pynisher.


Configuration Space
-------------------


Target Algorithm
----------------


Scenario
--------


Facade
------