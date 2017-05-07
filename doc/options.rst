SMAC-options and file-formats
-----------------------------
In the optimization-process of *SMAC*, there are several ways to configure the
options:

*Mandatory*:

* Commandline_-options, with which *SMAC* is called directly
* Scenario_-options, that are specified via a Scenario-object. Either directly
  in the Python-code or by using a scenario_-file.
* Parameter Configuration Space (PCS_), that provides the legal ranges of
  parameters to optimize, their types (e.g. int or float) and their default
  values.

*Optional*:

* Instance_- and feature_-files, that list the instances and features to
  optimize upon.

.. _commandline:

Commandline
~~~~~~~~~~~
*SMAC* is called via the command-line with the following arguments:
.. code-block:: bash

        python smac  - -scenario SCENARIO --seed INT --verbose_level LEVEL --modus MODUS

Required:
     * *scenario*: Path to the file that specifies the scenario_ for this *SMAC*-run.
Optional:
     * *seed*: The integer that the random-generator will be based upon. Default: 12345
     * *verbose_level*: in [INFO, DEBUG], specifies the logging-verbosity. Default: INFO
     * *modus*: in [SMAC, ROAR]. SMAC will use the bayeasian optimization with an intensification process, whereas ROAR stands for Random Online Adaptive Racing. Default: SMAC

.. _scenario:

Scenario-options
~~~~~~~~~~~~~~~~
The Scenario-object can be constructed either by providing an actual
Scenario-object (see `SVM <quickstart.html#using-smac-in-python-svm>`_-example), or by specifing the options in a
scenario file.
The format of the scenario file is one option per line:

.. code-block:: bash

        OPTION1 VALUE1
        OPTION2 VALUE2
        ...

For boolean options "1" or "true" both evaluate to True.

Required:
        * *algo* specifies the target-algorithm call that *SMAC* will optimize. Interpreted as a bash-command.
        * *paramfile* specifies the path to the PCS-file
        * *cutoff_time* is the maximum runtime, after which the target-algorithm is cancelled. **Required if *run_obj* is runtime.**

Optional:
        * *abort_on_first_run_crash* in [true, false]. If true, *SMAC* will abort if the first run of the target algorithm crashes. Default: true.
        * *execdir* specifies the path to the execution-directory. Default: ".".
        * *deterministic* in [true, false]. If true, the optimization process will be repeatable. Default: false 
        * *run_obj* in [runtime, quality]. Defines what metric to optimize. When optimizing runtime, *cutoff_time* is required as well. Default: runtime.
        * *overall_obj* is PARX, where X is an integer defining the penalty imposed on timeouts (i.e. runtimes that exceed the *cutoff-time*). Default: PAR10.
        * *memory_limit* is the maximum available memory the target-algorithm can occupy before being cancelled.
        * *tuner-timeout* is the maximum amount of CPU-time used for optimization. Default: inf.
        * *wallclock_limit* is the maximum amount of wallclock-time used for optimization. Default: inf.
        * *runcount_limit* is the maximum number of algorithm-calls during optimization. Default: inf.
        * *minR* is the minimum number of calls per configuration. Default: 1
        * *maxR* is the maximum number of calls per configuration. Default: 2000
        * *instance_file* specifies the file with the training-instances.
        * *test-instance_file* specifies the file with the test-instances.
        * *feature_file* specifies the file with the instance-features
        * *output_dir* specifies the output-directory for all emerging files, such as logging and results. Default: "smac3-output_YEAR-MONTH-DAY_HOUR:MINUTE:SECOND"
        * *shared_model*:  Default: false
        * *initial_incumbent*: in [DEFAULT, RANDOM]. DEFAULT is the default from the PCS. Default: DEFAULT.

.. _PCS:

Parameter Configuration Space (PCS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Parameter Configuration Space (PCS) defines the legal ranges of the
parameters to be optimized and their default values. In the examples-folder you
can find several examples for PCS-files. Generally, the format is:

.. code-block:: bash

        parameter_name categorical {value_1, ..., value_N} [default value]
        parameter_name ordinal {value_1, ..., value_N} [default value]
        parameter_name integer [min_value, max_value] [default value]
        parameter_name integer [min_value, max_value] [default value] log
        parameter_name real [min_value, max_value] [default value]
        parameter_name real [min_value, max_value] [default value] log

        # Conditionals:
        child_name | condition [&&,||] condition ...

        # Condition Operators: 
        # parent_x [<, >] parent_x_value (if parameter type is ordinal, integer or real)
        # parent_x [==,!=] parent_x_value (if parameter type is categorical, ordinal or integer)
        # parent_x in {parent_x_value1, parent_x_value2,...}

        # Forbiddens:
        {parameter_name_1=value_1, ..., parameter_name_N=value_N}

.. note::
        The PCS-format of *SMAC3* differs from that of the JAVA-based *SMAC2*.

.. _instance:
.. _feature:

Instances and Features
~~~~~~~~~~~~~~~~~~~~~~
To specify instances and features, simply provide text-files in the following
format and provide the paths to the instances in the scenario_.

Instance-files are text-files with one instance per line. If you want to use
training- and test-sets, separate files are expected.

Feature-files are files following the comma-separated-value-format, as can also be
seen in the `SPEAR-example <quickstart.html#spear-qcp>`_:

     +--------------------+--------------------+--------------------+-----+
     |      instance      | name of feature 1  | name of feature 2  | ... |
     +====================+====================+====================+=====+
     | name of instance 1 | value of feature 1 | value of feature 2 | ... |
     +--------------------+--------------------+--------------------+-----+
     |         ...        |          ...       |          ...       | ... |
     +--------------------+--------------------+--------------------+-----+
