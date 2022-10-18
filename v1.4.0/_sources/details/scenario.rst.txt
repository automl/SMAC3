Scenario
========

The scenario-object is used to configure SMAC and can be constructed either by providing an acutal
scenario-object or by specifing the options in a scenario file.

Create Scenario
~~~~~~~~~~~~~~~

Specify the options directly:

.. code-block:: python

    from smac.scenario.scenario import Scenario

    scenario = Scenario({
        'option': 'value',
        ...
    })

or specify the options in a scenario.txt file:

.. code-block:: 

    OPTION1 = VALUE1
    OPTION2 = VALUE2

Use ``"true"`` or ``"false"`` for boolean values, respectively.

Finally, create the scenario object by handing over the filename:

.. code-block:: python

    from smac.scenario.scenario import Scenario

    scenario = Scenario("/path/to/your/scenario.txt")


Options
~~~~~~~

Following options can be defined within the scenario object:

.. include:: ../scenario_options.rst

Additionally, you can also specify SMAC options with the scenario object:

.. include:: ../smac_options.rst

