F.A.Q.
======


Should I use SMAC2 or SMAC3?
  SMAC3 is a reimplementation of the original SMAC tool (`Sequential Model-Based Optimization for
  General Algorithm Configuration <https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/11-LION5-SMAC.pdf>`_, Hutter et al., 2021). However, the reimplementation slightly differs from the original
  SMAC. For comparisons against the original SMAC, we refer to a stable release of SMAC (v2) in Java
  which can be found `here <http://www.cs.ubc.ca/labs/beta/Projects/SMAC/>`_.
  Since SMAC3 is actively maintained, we recommend to use SMAC3 for any AutoML applications.


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


How can I use :term:`BOHB` and/or `HpBandSter <https://github.com/automl/HpBandSter>`_ with SMAC?
  The facade SMAC4HPO is the closes implementation to :term:`BOHB` and/or `HpBandSter <https://github.com/automl/HpBandSter>`_.


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


Why does SMAC not run on Colab/Mac and crashes with the error "Child process not yet created"?
  SMAC uses pynisher to enforce time and memory limits on the target algorithm runner. However, pynisher may not always
  work on specific setups. To overcome this error, it is recommended to set `limit_resources` to false to make SMAC run.
