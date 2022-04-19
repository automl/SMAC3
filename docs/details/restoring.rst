Restoring
=========

Python
~~~~~~

.. note::
    This chapter is under construction.


Commandline
~~~~~~~~~~~

If a SMAC run was interrupted or you want to extend its computation- or
time-limits, it can be restored and continued.
To restore or continue a previous SMAC run, use the
``--restore_state FOLDER`` option in the commandline. If you want to increase
computation- or time-limits, change the scenario-file specified with the
``--scenario SCENARIOFILE`` option (not the one in the folder to be restored).
Restarting a SMAC run that quit due to budget-exhaustion will do nothing,
because the budget is still exhausted.

.. warning::
    Changing any other options than ``output_dir``, ``wallclock_limit``, ``runcount_limit`` or
    ``tuner-timeout`` in the scenario-file is NOT intended and will likely lead
    to unexpected behaviour!

For an example of restoring states from within your Python code, there is an
implementation with the Branin-example in "examples/quickstart/branin/restore_state.py".