Basic Usage
-----------

There are two ways to use *SMAC*, either over the commandline or within your
Python-code.

Either way, you need to provide the `target algorithm`_ you want to
optimize and the `configuration space`_, which specifies the legal ranges and
default values of the tunable parameters. In addition, you can configure the
optimization process with the `scenario`_-options.

The most important scenario-options are:

- **run_obj**: Either quality or runtime, depending on what you want to
  optimize.
- **cutoff_time**: This is the maximum time the target algorithm is allowed to
  run.
- **wallclock_limit**, **runcount_limit** and **tuner-timeout**
  are used to control maximum wallclock-time, number of algorithm calls and
  cpu-time used for optimization respectively.
- **instance_file**, **test_instance_file** and **feature_file** specify the
  paths to the instances and features (see `file-formats`_)

For a full list, see `scenario-options`_.

.. _commandline:

Commandline 
~~~~~~~~~~~
To use *SMAC* via the commandline, you need a `scenario-file`_ and a `PCS-file`_.
The script to invoke *SMAC* is located in *scripts/smac*. Please see the
`branin`_-example to see how to use it.

*SMAC* is called via the commandline with the following arguments:

.. code-block:: bash

        python smac --scenario SCENARIO --seed INT --verbose_level LEVEL --modus MODUS

Required:
     * *scenario*: Path to the file that specifies the scenario_ for this *SMAC*-run.
Optional:
     * *seed*: The integer that the random-generator will be based upon. Default: 12345
     * *verbose_level*: in [INFO, DEBUG], specifies the logging-verbosity. Default: INFO
     * *modus*: in [SMAC, ROAR]. SMAC will use the bayeasian optimization with an intensification process, whereas ROAR stands for Random Online Adaptive Racing. Default: SMAC

In the scenario file, there are two mandatory parameters: The **algo**-parameter
defines how to call the target algorithm you want to optimize. *SMAC* will
execute the command and append the parameters to optimize to the call with
``-PARAMETER VALUE``, so make sure your algorithm will use the parameters in this
form. The **paramfile**-parameter defines the path to the `PCS-file`_. Both will
interpret paths *from the execution-directory*.

The `PCS-file <>`_ describes the ranges and default values of the tunable parameters.

.. note::

    Currently, running *SMAC* via the commandline will register the algorithm with a
    `Target Algorithm Evaluator (TAE) <>`_, that requires the target algorithm to print
    the results to the console in the following format (see `branin <>`_):
    
    .. code-block:: bash
    
        Result for SMAC: <STATUS>, <runtime>, <runlength>, <quality>, <seed>, <instance-specifics>


Usage in Python
~~~~~~~~~~~~~~~
The usage of *SMAC* from your Python-code is described in the `SVM-example`_.
Scenario and PCS are both build within the code. The target algorithm needs to
be registered with a `Target Algorithm Evaluator (TAE) <>`_, which communicates
between *SMAC* and the target algorithm. To optimize a function, you can instantiate
`ExecuteTAFuncDict <apidoc/smac.tae.html#smac.tae.execute_func.ExecuteTAFuncDict>`_ or 
`ExecuteTAFuncArray <apidoc/smac.tae.html#smac.tae.execute_func.ExecuteTAFuncArray>`_.
In that case, the algorithm needs to return a cost, representing the quality of
the solution, while time- and memory-limits are enforced by `Pynisher
<https://github.com/sfalkner/pynisher>`_, so no wrapper is needed for your
algorithm here.
