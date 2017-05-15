.. _branin: quickstart.html#branin
.. _svm-example: quickstart.html#svm-example
.. _spear: quickstart.html#spear

Target Algorithm Evaluator
--------------------------

*SMAC* evaluates the object to be optimized by invoking it through a Target
Algortihm Evaluator (TAE). There are two main distinctions, one is the TAE
used when invoking *SMAC* via the commandline, the other one is used when
optimizing functions within python. The main difference is that the former
interprets score and runtime from the output printed by the target algorithm,
the latter simply monitors time- and memory-usage and interprets the quality
from the target algorithms return value directly.

TAE-Commandline
~~~~~~~~~~~~~~~
*smac.tae.execute_ta_run_old*. Used when calling *SMAC* via the commandline_, where it takes the algorithm call
from the algo-parameter of the scenario_. It expects the target algorithm to print a string during
execution with the following format:

    .. code-block:: bash

        Result for SMAC: <STATUS>, <runtime>, <runlength>, <quality>, <seed>, <instance-specifics>

The examples branin_ or SPEAR_ reflect the usage.

* **STATUS** can be one of [SAT, UNSAT, SUCCESS, TIMEOUT, MEMOUT, CRASHED, ABORT]. SAT and UNSAT are mainly supported for backcompatibility and are treated as SUCCESS. The difference between CRASHED and ABORT is that ABORT is called when all future calls are assumed to crash and will abort the whole optimization, whereas CRASHED only indicates a single failed run.
* **runtime** indicates the time that the execution took.
* **runlength** indicates the number of steps needed for execution.
* **quality** is the solution quality.
* **seed** is the seed used for the algorithm call.
* **instance-specifics** is additional information.

TAE-Function
~~~~~~~~~~~~
*smac.tae.execute_func*.
You can register any function returning a loss/score to a function
evaluator. This TAE uses `pynisher <https://github.com/sfalkner/pynisher>`_ to
enforce time- and memorylimits. There is no need for the target algorithm to print a
string, instead the score is returned after execution and interpreted by the
TAE. This can be seen in the `svm-example`_.

