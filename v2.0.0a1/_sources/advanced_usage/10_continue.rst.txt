Continue a Run
==============

SMAC automatically restores states where it left off if a run was interrupted or finished. To do so, it reads in 
files (derivided from scenario's name, output_directory and seed) and sets the components.

.. warning::

    If you changed any code and specified a name, SMAC will ask you whether you still want to resume or
    delete the old run completely. If you not specified a name, SMAC generates a new name and the old run is
    not affected.

Unfortunately, since many components of SMAC have internal states (especially the intensifier), it is not possible to
continue a run from a previous state yet.

