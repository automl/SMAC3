F.A.Q.
======


SMAC cannot be imported.
  Try to either run SMAC from SMAC's root directory
  or try to run the installation first.


pyrfr raises cryptic import errors.
  Ensure that the gcc used to compile the pyrfr is the same as used for linking
  during execution. This often happens with Anaconda. See
  :ref:`Installation <installation>` for a solution.


My target algorithm is not accepted when using the scenario-file.
  Make sure that your algorithm accepts commandline options as provided by
  SMAC. Refer to :ref:`commandline execution <Basic Usage>` for
  details on how to wrap your algorithm.

  You can also run SMAC with ``--verbose DEBUG`` to see how SMAC tried to call your algorithm.


Can I restore SMAC from a previous state?
  Yes. Have a look :ref:`here<Restoring>`.


I discovered a bug or SMAC does not behave as expected. Where should I report to?
  Open an issue in our issue list on GitHub. Before you report a bug, please make sure that:

  * Your bug hasn't already been reported in our issue tracker.
  * You are using the latest SMAC3 version.

  If you found an issue, please provide us with the following information:

  * A description of the problem.
  * An example to reproduce the problem.
  * Any information about your setup that could be helpful to resolve the bug (such as installed python packages).
  * Feel free, to add a screenshot showing the issue.


I want to contribute code or discuss a new idea. Where should I report to?
  SMAC uses the `GitHub issue-tracker <https://github.com/automl/SMAC3/issues>`_ to also take care
  of questions and feedback and is the preferred location for discussing new features and ongoing work. Please also have a look at our
  `contribution guide <https://github.com/automl/SMAC3/blob/master/CONTRIBUTING.md>`_.


What is the meaning of *deterministic*?
  If the ``deterministic`` flag is set to `False` the target algorithm is assumed to be non-deterministic.
  To evaluate a configuration of a non-deterministic algorithm, multiple runs with different seeds will be evaluated
  to determine the performance of that configuration on one instance.
  Deterministic algorithms don't depend on seeds, thus requiring only one evaluation of a configuration on an instance
  to evaluate the performance on that instance. Nevertheless the default seed 0 is still passed to the
  target algorithm.


I want my algorithm to be optimized across different datasets. How should I realize that?
  Generally, you have two options: Validate all datasets within your :ref:`TAE<Target Algorithm Evaluator>` or use instances.
  The significant advantage of instances is that not all datasets necessarily have to be processed.
  If the first instances already perform worse, the configuration might be discarded early. This
  will lead to a speed-up.


