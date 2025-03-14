# F.A.Q.

#### Should I use SMAC2 or SMAC3?
  SMAC3 is a reimplementation of the original SMAC tool ([Sequential Model-Based Optimization for General Algorithm Configuration](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/11-LION5-SMAC.pdf), Hutter et al., 2021). However, the reimplementation slightly differs from the original
  SMAC. For comparisons against the original SMAC, we refer to a stable release of SMAC (v2) in Java
  which can be found [here](https://www.cs.ubc.ca/labs/algorithms/Projects/SMAC/).
  Since SMAC3 is actively maintained, we recommend to use SMAC3 for any AutoML applications.


#### SMAC cannot be imported.
  Try to either run SMAC from SMAC's root directory or try to run the installation first.


#### pyrfr raises cryptic import errors.
  Ensure that the gcc used to compile the pyrfr is the same as used for linking
  during execution. This often happens with Anaconda. See
  [Installation](1_installation.md) for a solution.


#### How can I use :term:`BOHB` and/or [HpBandSter](https://github.com/automl/HpBandSter) with SMAC?
  The facade MultiFidelityFacade is the closest implementation to :term:`BOHB` and/or [HpBandSter](https://github.com/automl/HpBandSter).


#### I discovered a bug or SMAC does not behave as expected. Where should I report to?
  Open an issue in our issue list on GitHub. Before you report a bug, please make sure that:

  * Your bug hasn't already been reported in our issue tracker.
  * You are using the latest SMAC3 version.

  If you found an issue, please provide us with the following information:

  * A description of the problem.
  * An example to reproduce the problem.
  * Any information about your setup that could be helpful to resolve the bug (such as installed python packages).
  * Feel free to add a screenshot showing the issue.


#### I want to contribute code or discuss a new idea. Where should I report to?
  SMAC uses the [GitHub issue-tracker](https://github.com/automl/SMAC3/issues) to also take care
  of questions and feedback and is the preferred location for discussing new features and ongoing work. Please also have a look at our
  [contribution guide](https://github.com/automl/SMAC3/blob/main/CONTRIBUTING.md).


#### What is the meaning of *deterministic*?
  If the ``deterministic`` flag is set to `False` the target function is assumed to be non-deterministic.
  To evaluate a configuration of a non-deterministic algorithm, multiple runs with different seeds will be evaluated
  to determine the performance of that configuration on one instance.
  Deterministic algorithms don't depend on seeds, thus requiring only one evaluation of a configuration on an instance
  to evaluate the performance on that instance. Nevertheless the default seed 0 is still passed to the
  target function.


#### Why does SMAC not run on Colab/Mac and crashes with the error "Child process not yet created"?
  SMAC uses pynisher to enforce time and memory limits on the target function runner. However, pynisher may not always
  work on specific setups. To overcome this error, it is recommended to remove limitations to make SMAC run.
