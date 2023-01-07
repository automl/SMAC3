Logging
=======

Logging is a crucial part of the optimization, which should be customizable by the user. This page gives you the
overview how to customize the logging experience with SMAC.

Level
-----

The easiest way to change the logging behaviour is to change the level of the global logger. SMAC does this for you
if you specify the ``logging_level`` in any facade.

.. code-block:: python 

    smac = Facade(
        ...
        logging_level=20,
        ...
    )


The table shows you the specific levels:

.. csv-table::
    :header: "Name", "Level"

    0, SHOW ALL
    10, DEBUG
    20, INFO 
    30, WARNING
    40, ERROR 
    50, CRITICAL


Custom File
-----------

Sometimes, the user wants to disable or highlight specify modules. You can do that by passing a custom yaml
file to the facade instead.

.. code-block:: python 

    smac = Facade(
        ...
        logging_level="path/to/your/logging.yaml",
        ...
    )


The following file shows you how to display only error messages from the intensifier 
but keep the level of everything else on INFO:

.. code-block:: yaml

    version: 1
    disable_existing_loggers: false
    formatters:
        simple:
            format: '[%(levelname)s][%(filename)s:%(lineno)d] %(message)s'
    handlers:
        console:
            class: logging.StreamHandler
            level: INFO
            formatter: simple
            stream: ext://sys.stdout
    loggers:
        smac.intensifier:
            level: ERROR
            handlers: [console]
    root:
        level: INFO
        handlers: [console]

