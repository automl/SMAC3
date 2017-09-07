Options and file formats
------------------------
In the optimization-process of *SMAC*, there are several ways to configure the
options:

Mandatory:
    * `Commandline <basic_usage.html#commandline>`_-options, with which *SMAC* is called directly (not needed if
      *SMAC* is used within Python).
    * Scenario-options, that are specified via a Scenario-object. Either directly
      in the Python-code or by using a scenario_-file.
    * A Parameter Configuration Space (`PCS <options.html#paramcs>`_), that provides the legal ranges of
      parameters to optimize, their types (e.g. int or float) and their default
      values.

Optional:
    * Instance_- and feature_-files, that list the instances and features to
      optimize upon.

.. _scenario:

Scenario
~~~~~~~~
The scenario-object (:class:`smac.scenario.scenario.Scenario`) is used to configure *SMAC* and can be constructed either by providing an actual
scenario-object (see `SVM-example <quickstart.html#using-smac-in-python-svm>`_), or by specifing the options in a
scenario file (see `SPEAR example <quickstart.html#spear-example>`_).
The format of the scenario file is one option per line:

.. code-block:: bash

        OPTION1 = VALUE1
        OPTION2 = VALUE2
        ...

For boolean options "1" or "true" both evaluate to True.
The following assumes that the scenario is created via a scenario-file. If it is
generated within custom code, you might not need *algo* or *paramfile*.

Options:

.. include:: scenario_options.rst

.. _paramcs:

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
