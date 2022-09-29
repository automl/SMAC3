Continue a Run
==============

SMAC automatically resumes where it left off if it was interrupted. To do so, it reads in files
(derivided from scenario's name, output_directory and seed) and sets the internal states.

.. warning::

    If you changed any code and specified a name, SMAC will ask you whether you still want to resume or
    delete the old run completely. If you not specified a name, SMAC generates a new name and the old run is
    not affected.

