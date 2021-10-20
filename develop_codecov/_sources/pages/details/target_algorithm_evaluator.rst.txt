Target Algorithm Evaluator
==========================

SMAC evaluates the algorithm to be optimized by invoking it through a Target
Algorithm Evaluator (TAE). There are different TAEs implemented in SMAC which
provide slightly different interfaces.


Python
------

The target algorithm is called with a dict- or array-like configuration and optionally
with seed and instance, returning either the loss as a float or a tuple (loss,
additional information). This is very handy when used within Python to
optimize any blackbox-function. Using this TAE, there is no need to print a
result string; the error will be interpreted from the return object. With this TAE, `Pynisher <https://github.com/sfalkner/pynisher>`_ is used to
enforce time- and memory-limits.

.. warning::

  SMAC is always minimizing, meaning that smaller values are better. For example, if you want to
  optimize accuracy, you have to use ``1 - accuracy`` for correct usage.

The following example shows the interface for all possible arguments:

.. code-block:: python

  train_and_validate_your_model(config, budget, instance, seed):
    # ...
    return 1 - accuracy


If you want to return additional information, which will be written into the RunHistory, you can
follow the next example:

.. code-block:: python

    train_and_validate_your_model(config, budget, instance, seed):
      # ...
      additional_info = {
        "loss": loss,
        "something_else": ...
      }
      return (1 - accuracy, additional_info)

Finally, pass your function to your SMAC facade:

.. code-block:: python

  # Create your SMAC object
  smac = SMAC4BB(
    ...
    tae_runner=train_and_validate_your_model
    ...
  )



Commandline
-----------

When using the commandline, SMAC takes the algorithm call from the ``algo`` parameter of the
:ref:`scenario<Scenario>`. The parameters will be appended to the algorithm call, which in total
looks like:

.. code-block:: bash

  <algo> <instance> <instance specific> <cutoff time> <runlength> <seed> <algorithm parameters>
  python3 algo.py 123 0 10 25 12345 -param1 value1 -param2 value2 [...]

The first two parameters after the ``algo.py`` are the instance name, on which the target algorithm is evaluated, and extra information about the instance (rarely used).
The third parameter is the cutoff time, which is the maximal time the target algorithm is allowed to run.
The fourth parameter is the runlength, which is the maximal number of steps an algorithm is allowed to run.
The fifth parameter is the random seed which is followed by the target algorithm parameters.

It expects the target algorithm to print a string during execution with on of the following formats:

.. code-block:: bash

  Result of this algorithm run: <STATUS>, <running time>, <runlength>, <quality>, <seed>, <instance-specifics>
  Result for SMAC: <STATUS>, <running time>, <runlength>, <quality>, <seed>, <instance-specifics>
  Result for ParamILS: <STATUS>, <running time>, <runlength>, <quality>, <seed>, <instance-specifics>

The example :ref:`Branin` reflect the usage.

* **STATUS** can be one of [SAT, UNSAT, SUCCESS, TIMEOUT, MEMOUT, CRASHED, ABORT]. SAT and UNSAT are mainly supported for backcompatibility and are treated as SUCCESS. The difference between CRASHED and ABORT is that ABORT is called when all future calls are assumed to crash and will abort the whole optimization, whereas CRASHED only indicates a single failed run.
* **running time** indicates the time that the execution took.
* **runlength** indicates the number of steps needed for execution.
* **quality** is the solution quality.
* **seed** is the seed used for the algorithm call.
* **instance-specifics** is additional information.
