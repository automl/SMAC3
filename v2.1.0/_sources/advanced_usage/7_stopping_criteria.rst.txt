Stopping Criteria
=================

In addition to the standard stopping criteria like number of trials or wallclock time, SMAC also provides 
more advanced criteria.


Termination Cost Threshold
--------------------------

SMAC can stop the optimization process after a user-defined cost was reached. In each iteration, the average cost 
(using ``average_cost`` from the run history) from the incumbent is compared to the termination cost threshold. If one
of the objective costs is below its associated termination cost threshold, the optimization process is stopped.
Note, since the ``average_cost`` method is used, all instance-seed-budget trials of the incumbent are considered so far.
In other words, the process can be stopped even if the incumbent has not been evaluated on all instances, on the 
highest fidelity, or on all seeds.


.. code-block:: python

    scenario = Scenario(
        ...
        objectives=["accuracy", "runtime"],
        termination_cost_threshold=[0.1, np.inf]
        ...
    )

In the code above, the optimization process is stopped if the average accuracy of the incumbent is below 0.1. The 
runtime is ignored completely as it is set to infinity. Note here again that SMAC minimizes the objective values.


Automatically Stopping
----------------------

Coming in the next version of SMAC.