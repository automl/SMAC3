# Continue

SMAC can automatically restore states where it left off if a run was interrupted or prematurely finished. To do so, 
it reads in old files (derived from scenario's name, output_directory and seed) and obtains the scenario information
of the previous run from those to continue the run.

The behavior can be controlled by setting the parameter ``continue_run`` in the Scenario to False or True, respectively:

* If set to False (default), SMAC overwrites the run results if a previous run is found that is consistent in the meta data with the current setup.
* If set to True and a previous run is found that

  * is consistent in the meta data, the run is continued. 
  * is not consistent in the meta data, the user is asked for the exact behaviour (overwrite completely or rename old run first).

!!! warning
    If you changed any code affecting the run's meta data and specified a name, SMAC will ask you whether you still 
    want to overwrite the old run or rename the old run first. If you did not specify a name, SMAC generates a new name 
    and the old run is not affected.

The `continue_run` parameter can also be set with `Scenario.set_continue_run(continue_run: bool)` after instantiation of Scenario.

!!! warning "Deprecated"
    The `overwrite` flag of the Facade has been replaced by `continue_run` in the Scenario, which has the opposite meaning (`continue_run = not overwrite`). For compatibility, setting `overwrite` in the Facade still works and takes precedence over `Scenario.continue_run`, but it shows a deprecation warning and will be removed in the future.

Please have a look at our [continue example](../examples/1%20Basics/5_continue.md).