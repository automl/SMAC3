F.A.Q.
======

.. rubric:: The examples won't work as expected.

Try to adjust your execution directory, since examples like svm.py and rf.py are
using *SMAC* within Python and have to be executed from the *SMAC*
root-directory.

.. rubric:: My target algorithm is not accepted, when using the scenario-file.

Make sure that your algorithm accepts commandline options as provided by *SMAC*.
Refer to `commandline execution <basic_usage.html#commandline>`_ for details on how to wrap your algorithm.

.. rubric:: I discovered a bug/have criticism or ideas on *SMAC*. Where should I report to?

*SMAC* uses the GitHub issue-tracker to take care of bugs and questions. If you
experience problems with *SMAC*, try to provide a full error report with all the
typical information (OS, version, console-output, minimum working example, ...).
This makes it a lot easier to reproduce the error and locate the problem.


.. rubric:: **Glossary**

* **SMAC**: Sequential Model-Based Algorithm Configuration
* **ROAR**: Random Online Adaptive Racing
* **PCS**: Parameter Configuration Space
* **TAE**: Target Algorithm Evaluator

