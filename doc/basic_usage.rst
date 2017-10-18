Basic Usage
-----------

There are two ways to use *SMAC*, either over the commandline or within your
Python-code.

Either way, you need to provide the `target algorithm <tae.html#tae>`_ you want to
optimize and the `configuration space <options.html#pcs>`_, which specifies the legal ranges and
default values of the tunable parameters. In addition, you can configure the
optimization process with the `scenario <options.html#scenario>`_-options.

The most important scenario-options are:

- **run_obj**: Either quality or runtime, depending on what you want to
  optimize.
- **cutoff_time**: This is the maximum time the target algorithm is allowed to
  run.
- **wallclock_limit**, **runcount_limit** and **tuner-timeout**
  are used to control maximum wallclock-time, number of algorithm calls and
  cpu-time used for optimization respectively.
- **instance_file**, **test_instance_file** and **feature_file** specify the
  paths to the instances and features (see `file-formats <options.html#instance>`_)

For a full list, see `scenario-options <options.html#scenario>`_.

.. _commandline:

Commandline 
~~~~~~~~~~~
To use *SMAC* via the commandline, you need a `scenario-file <options.html#scenario>`_ and a `PCS-file <options.html#pcs>`_.
The script to invoke *SMAC* is located in *scripts/smac*. Please see the
`Branin <quickstart.html#branin>`_-example to see how to use it.

*SMAC* is called via the commandline with the following arguments:

.. code-block:: bash

        python smac --scenario SCENARIO --seed INT --verbose_level LEVEL --mode MODE

Required:
     * *scenario*: Path to the file that specifies the `scenario <options.html#scenario>`_ for this *SMAC*-run.
Optional:
     * *seed*: The integer that the random-generator will be based upon. **Default**: 12345
     * *verbose_level*: in [INFO, DEBUG], specifies the logging-verbosity. **Default**: INFO
     * *mode*: in [SMAC, ROAR]. SMAC will use the bayeasian optimization with an intensification process, whereas ROAR stands for Random Online Adaptive Racing. **Default**: SMAC
     * *restore_state*: A string specifying the folder of the *SMAC*-run to be continued. **Assuming exactly the same scenario, except for budget-options.**

In the scenario file, there are two mandatory parameters: The **algo**-parameter
defines how *SMAC* will call the target algorithm. Parameters will be appended to the call
with ``-PARAMETER VALUE``, so make sure your algorithm will accept the parameters in this
form. Read more in the section on `target algorithms <tae.html#tae>`_.
The **paramfile**-parameter defines the path to the `PCS-file <options.html#pcs>`_,
which describes the ranges and default values of the tunable parameters.
Both will interpret paths *from the execution-directory*.

.. note::

    Currently, running *SMAC* via the commandline will register the algorithm with a
    `Target Algorithm Evaluator (TAE) <tae.html#tae>`_, that requires the target algorithm to print
    the results to the console in the following format (see `Branin
    <quickstart.html#branin>`_):
    
    .. code-block:: bash
    
        Result for SMAC: <STATUS>, <runtime>, <runlength>, <quality>, <seed>, <instance-specifics>


.. _restorestate:

Restoring States
~~~~~~~~~~~~~~~~
If a *SMAC*-run was interrupted or you want to extend its computation- or
time-limits, it can be restored and continued.
To restore or continue a previous *SMAC*-run, use the
``--restore_state FOLDER``-option in the commandline. If you want to increase
computation- or time-limits, change the scenario-file specified with the
``--scenario SCENARIOFILE``-option (**not the one in the folder to be restored**).
Restarting a *SMAC*-run that quit due to budget-exhaustion will do nothing,
because the budget is still exhausted.
**Changing any other options than *output_dir*, *wallclock_limit*, *runcount_limit* or
*tuner-timeout* in the scenario-file is NOT intended and will likely lead
to unexpected behaviour!**

For an example of restoring states from within your Python code, there is an
implementation with the Branin-example in "examples/branin/restore_state.py".


.. _inpython:

Usage in Python
~~~~~~~~~~~~~~~
The usage of *SMAC* from your Python-code is described in the `SVM-example
<quickstart.html#svm-example>`_.
Scenario and configuration space are both build within the code. The target
algorithm needs to be registered with a `Target Algorithm Evaluator (TAE) <tae.html#tae>`_,
which communicates between *SMAC* and the target algorithm. To optimize a function, you can instantiate
`ExecuteTAFuncDict <apidoc/smac.tae.execute_func.html#smac.tae.execute_func.ExecuteTAFuncDict>`_ or
`ExecuteTAFuncArray <apidoc/smac.tae.execute_func.html#smac.tae.execute_func.ExecuteTAFuncArray>`_.
In that case, the algorithm needs to return a cost, representing the quality of
the solution, while time- and memory-limits are measured and enforced by `Pynisher
<https://github.com/sfalkner/pynisher>`_, so no wrapper is needed for your
algorithm here.

- `ExecuteTAFuncDict <apidoc/smac.tae.execute_func.html#smac.tae.execute_func.ExecuteTAFuncDict>`_:
  The target algorithm is called with a dict-like configuration and optionally
  with seed and instance, returning either the loss as a float or a tuple (loss,
  additional information).
- `ExecuteTAFuncArray <apidoc/smac.tae.execute_func.html#smac.tae.execute_func.ExecuteTAFuncArray>`_:
  The target algorithm is called with an array-like configuration and optionally
  with seed and instance, returning either the loss as a float or a tuple (loss,
  additional information).

