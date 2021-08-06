Basic Usage
-----------

There are two ways to use *SMAC*, either over the commandline or within your
Python-code.

Either way, you need to provide the :ref:`target algorithm <tae>` you want to
optimize and the :ref:`configuration space <paramcs>`, which specifies the legal ranges and
default values of the tunable parameters. In addition, you can configure the
optimization process with the :ref:`scenario <scenario>`-options.

The most important scenario-options are:

- **run_obj**: Either quality or runtime, depending on what you want to
  optimize.
- **cutoff_time**: This is the maximum time the target algorithm is allowed to
  run.
- **wallclock_limit**, **runcount_limit** and **tuner-timeout**
  are used to control maximum wallclock-time, number of algorithm calls and
  cpu-time used for optimization respectively.
- **instance_file**, **test_instance_file** and **feature_file** specify the
  paths to the instances and features (see :ref:`file-formats <instance>`)

For a full list, see :ref:`scenario-options <scenario>`.

.. _commandline:

Commandline 
~~~~~~~~~~~
To use *SMAC* via the commandline, you need a :ref:`scenario-file <scenario>` and a :ref:`PCS-file <paramcs>`.
The script to invoke *SMAC* is located in *scripts/smac*. Please see the
:ref:`Branin <branin-example>`-example to see how to use it.

*SMAC* is called via the commandline with the following arguments:

.. code-block:: bash

        python3 smac --scenario SCENARIO --seed INT --verbose_level LEVEL --mode MODE

Required:
     * *scenario*: Path to the file that specifies the `scenario <scenario>` for this *SMAC*-run.
Optional:
     * *seed*: The integer that the random-generator will be based upon. **Default**: 12345
     * *verbose_level*: in [INFO, DEBUG], specifies the logging-verbosity. **Default**: INFO
     * *mode*: in [SMAC, ROAR]. SMAC will use the bayeasian optimization with an intensification process, whereas ROAR stands for Random Online Adaptive Racing. **Default**: SMAC
     * *restore_state*: A string specifying the folder of the *SMAC*-run to be continued. **Assuming exactly the same scenario, except for budget-options.**

In the scenario file, there are two mandatory parameters: The **algo**-parameter
defines how *SMAC* will call the target algorithm. Parameters will be appended to the call
with ``-PARAMETER VALUE``, so make sure your algorithm will accept the parameters in this
form. Read more in the section on :ref:`target algorithms <tae>`.
The **paramfile**-parameter defines the path to the `PCS-file <paramcs>`,
which describes the ranges and default values of the tunable parameters.
Both will interpret paths *from the execution-directory*.

.. note::

    Currently, running *SMAC* via the commandline will register the algorithm with a
    :ref:`Target Algorithm Evaluator (TAE) <tae>`, that requires the target algorithm to print
    the results to the console in the following format (see :ref:`Branin <branin-example>`):
    
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
implementation with the Branin-example in "examples/quickstart/branin/restore_state.py".


.. _inpython:

Usage in Python
~~~~~~~~~~~~~~~
The usage of *SMAC* from your Python-code is described in the :ref:`SVM-example <svm-example>`.
Scenario and configuration space are both build within the code. The target
algorithm needs to be registered with a :ref:`Target Algorithm Evaluator (TAE) <tae>`,
which communicates between *SMAC* and the target algorithm. To optimize a function, you can instantiate
:class:`ExecuteTAFuncDict <smac.tae.execute_func.ExecuteTAFuncDict>` or
:class:`ExecuteTAFuncArray <smac.tae.execute_func.ExecuteTAFuncArray>`.
In that case, the algorithm needs to return a cost, representing the quality of
the solution, while time- and memory-limits are measured and enforced by `Pynisher
<https://github.com/sfalkner/pynisher>`_, so no wrapper is needed for your
algorithm here.

- :class:`ExecuteTAFuncDict <smac.tae.execute_func.ExecuteTAFuncDict>`:
  The target algorithm is called with a dict-like configuration and optionally
  with seed and instance, returning either the loss as a float or a tuple (loss,
  additional information).
- :class:`ExecuteTAFuncArray <smac.tae.execute_func.ExecuteTAFuncArray>`:
  The target algorithm is called with an array-like configuration and optionally
  with seed and instance, returning either the loss as a float or a tuple (loss,
  additional information).

