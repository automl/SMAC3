Continue
========

SMAC can automatically restore states where it left off if a run was interrupted or prematurely finished. To do so, 
it reads in old files (derived from scenario's name, output_directory and seed) and obtains the scenario information
of the previous run from those to continue the run.

The behavior can be controlled by setting the parameter ``overwrite`` in the facade to True or False, respectively:

* If set to True, SMAC overwrites the run results if a previous run is found that is consistent in the meta data with the current setup.
* If set to False and a previous run is found that

  * is consistent in the meta data, the run is continued. 
  * is not consistent in the meta data, the user is asked for the exact behaviour (overwrite completely or rename old run first).

.. warning::

    If you changed any code affecting the run's meta data and specified a name, SMAC will ask you whether you still 
    want to overwrite the old run or rename the old run first. If you did not specify a name, SMAC generates a new name 
    and the old run is not affected.


Please have a look at our :ref:`continue example<Continue an Optimization>`.