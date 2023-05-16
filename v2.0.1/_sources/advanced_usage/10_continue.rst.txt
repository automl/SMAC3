Continue
========

SMAC automatically restores states where it left off if a run was interrupted or finished. To do so, it reads in old
files (derived from scenario's name, output_directory and seed) and sets the components.

.. warning::

    If you changed any code and specified a name, SMAC will ask you whether you still want to resume or
    delete the old run completely. If you did not specify a name, SMAC generates a new name and the old run is
    not affected.


Please have a look at our :ref:`continue example<Continue an Optimization>`.