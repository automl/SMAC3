F.A.Q.
======

.. rubric:: SMAC cannot be imported.

Try to either run SMAC from SMAC's root directory
or try to run the installation first.

.. rubric:: pyrfr raises cryptic import errors.

Ensure that the gcc used to compile the pyrfr is the same as used for linking
during execution. This often happens with Anaconda -- see
`Installation <installation.html>`_ for a solution.

.. rubric:: My target algorithm is not accepted, when using the scenario-file.

Make sure that your algorithm accepts commandline options as provided by
*SMAC*. Refer to `commandline execution <basic_usage.html#commandline>`_ for
details on how to wrap your algorithm.

You can also run SMAC with :code:`--verbose DEBUG` to see how *SMAC* tried to call your algorithm.

.. rubric:: Can I restore SMAC from a previous state?

Use the `restore-option <basic_usage.html#restorestate>`_.

.. rubric:: I discovered a bug/have criticism or ideas on *SMAC*. Where should
I report to?

*SMAC* uses the
`GitHub issue-tracker <https://github.com/automl/SMAC3/issues>`_ to take care
of bugs and questions. If you experience problems with *SMAC*, try to provide
a full error report with all the typical information (OS, version,
console-output, minimum working example, ...). This makes it a lot easier to
reproduce the error and locate the problem.


.. rubric:: **Glossary**

* **SMAC**: Sequential Model-Based Algorithm Configuration
* **ROAR**: Random Online Adaptive Racing
* **PCS**: Parameter Configuration Space
* **TAE**: Target Algorithm Evaluator

