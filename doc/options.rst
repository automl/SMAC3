Options and file formats
------------------------
In the optimization-process of *SMAC*, there are several ways to configure the
options:

Mandatory:
    * :ref:`Command line <commandline>`-options, with which *SMAC* is called directly (not needed if
      *SMAC* is used within Python).
    * Scenario-options, that are specified via a Scenario-object. Either directly
      in the Python-code or by using a :ref:`scenario <scenario>`-file.
    * A Parameter Configuration Space (:ref:`PCS <paramcs>`), that provides the legal ranges of
      parameters to optimize, their types (e.g. int or float) and their default
      values.

Optional:
    * :ref:`Instance <instance>`- and :ref:`feature <feature>`-files, that list the instances and features to
      optimize upon.

.. _smac_options:

SMAC Options
~~~~~~~~~~~~
The basic command line options are described in :ref:`Basic Usage <commandline>`.
The options are separated into three groups, *Main Options*, *SMAC Options* and *Scenario Options*.
See the Main and SMAC Options below. Find the Scenario Options in the next section.

Main Options:

.. include:: main_options.rst

SMAC Options:

.. include:: smac_options.rst

.. _scenario:

Scenario
~~~~~~~~
The scenario-object (:class:`smac.scenario.scenario.Scenario`) is used to configure *SMAC* and can be constructed either by providing an actual
scenario-object (see :ref:`SVM-example <svm-example>`), or by specifing the options in a
scenario file (see :ref:`SPEAR example <spear-example>`).

The format of the scenario file is one option per line:

.. code-block:: bash

        OPTION1 = VALUE1
        OPTION2 = VALUE2
        ...

For boolean options "1" or "true" both evaluate to True.
The following assumes that the scenario is created via a scenario-file. If it is
generated within custom code, you might not need *algo* or *paramfile*.

Scenario Options:

.. include:: scenario_options.rst

These options are also available as command line switches: Prepend two "-" and replace each "_" by "-",
e.g. "wallclock_limit" becomes "--wallclock-limit". The options on the command line overwrite the values
given in the scenario file.

.. _paramcs:

Parameter Configuration Space (PCS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Parameter Configuration Space (PCS) defines the legal ranges of the
parameters to be optimized and their default values. In the examples-folder you
can find several examples for PCS-files. Generally, the format is:

To define parameters and their ranges, the following format is supported:

.. code-block:: bash

        parameter_name categorical {value_1, ..., value_N} [default value]
        parameter_name ordinal {value_1, ..., value_N} [default value]
        parameter_name integer [min_value, max_value] [default value]
        parameter_name integer [min_value, max_value] [default value] log
        parameter_name real [min_value, max_value] [default value]
        parameter_name real [min_value, max_value] [default value] log

The trailing "log" indicates that SMAC should sample from the defined ranges
on a log scale.

Furthermore, conditional dependencies can be expressed. That is useful if
a parameter activates sub-parameters. For example, only if a certain heuristic
is used, the heuristic's parameter are active and otherwise SMAC can ignore these.

.. code-block:: bash

        # Conditionals:
        child_name | condition [&&,||] condition ...

        # Condition Operators:
        # parent_x [<, >] parent_x_value (if parameter type is ordinal, integer or real)
        # parent_x [==,!=] parent_x_value (if parameter type is categorical, ordinal or integer)
        # parent_x in {parent_x_value1, parent_x_value2,...}

Forbidden constraints allow for specifications of forbidden combinations of
parameter values. Please note that SMAC uses a simple rejection sampling
strategy. Therefore, SMAC cannot handle efficiently highly constrained spaces.

.. code-block:: bash

        # Forbiddens:
        {parameter_name_1=value_1, ..., parameter_name_N=value_N}


.. _instance:
.. _feature:

Instances and Features
~~~~~~~~~~~~~~~~~~~~~~
To specify instances and features, simply provide text-files in the following
format and provide the paths to the instances in the :ref:`scenario <scenario>`.

Instance-files are text-files with one instance per line. If you want to use
training- and test-sets, separate files are expected.

Feature-files are files following the comma-separated-value-format, as can also be
seen in the :ref:`SPEAR-example <spear-example>`:

     +--------------------+--------------------+--------------------+-----+
     |      instance      | name of feature 1  | name of feature 2  | ... |
     +====================+====================+====================+=====+
     | name of instance 1 | value of feature 1 | value of feature 2 | ... |
     +--------------------+--------------------+--------------------+-----+
     |         ...        |          ...       |          ...       | ... |
     +--------------------+--------------------+--------------------+-----+
