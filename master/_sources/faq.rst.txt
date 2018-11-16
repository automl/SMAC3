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

.. rubric:: I discovered a bug or SMAC does not behave as expected. Where should I report to?

Open an issue in our issue list on GitHub. Before you report a bug, please make sure that:

  * Your bug hasn't already been reported in our issue tracker
  * You are using the latest SMAC3 version.

If you found an issue, please provide us with the following information:

  * A description of the problem
  * An example to reproduce the problem
  * Any information about your setup that could be helpful to resolve the bug (such as installed python packages)
  * Feel free, to add a screenshot showing the issue

.. rubric:: I want to contribute code or discuss a new idea. Where should I report to?

*SMAC* uses the `GitHub issue-tracker <https://github.com/automl/SMAC3/issues>`_ to also take care
of questions and feedback and is the preferred location for discussing new features and ongoing work. Please also have a look at our
`contribution guide <https://github.com/automl/SMAC3/blob/master/.github/CONTRIBUTING.md>`_.

.. rubric:: What is the meaning of *deterministic*?

If the *deterministic* flag is set to **False** the target algorithm is assumed to be non-deterministic.
To evaluate a configuration of a non-deterministic algorithm, multiple runs with different seeds will be evaluated
to determine the performance of that configuration on one instance.
Deterministic algorithms don't depend on seeds, thus requiring only one evaluation of a configuration on an instance
to evaluate the performance on that instance. Nevertheless the default seed 0 is still passed to the target algorithm.

.. rubric:: **Glossary**

* **SMAC**: Sequential Model-Based Algorithm Configuration
* **ROAR**: Random Online Adaptive Racing
* **PCS**: Parameter Configuration Space
* **TAE**: Target Algorithm Evaluator

