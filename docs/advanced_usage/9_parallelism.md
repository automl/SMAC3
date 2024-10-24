# Parallelism

SMAC supports multiple workers natively via Dask. Just specify ``n_workers`` in the scenario and you are ready to go. 


!!! note
    
    Please keep in mind that additional workers are only used to evaluate trials. The main thread still orchestrates the
    optimization process, including training the surrogate model.


!!! warning

    Using high number of workers when the target function evaluation is fast might be counterproductive due to the 
    overhead of communcation. Consider using only one worker in this case.


!!! warning

    When using multiple workers, SMAC is not reproducible anymore.


## Running on a Cluster

You can also pass a custom dask client, e.g. to run on a slurm cluster.
See our [parallelism example](../../examples/1%20Basics/7_parallelization_cluster).

!!! warning

    On some clusters you cannot spawn new jobs when running a SLURMCluster inside a
    job instead of on the login node. No obvious errors might be raised but it can hang silently.

!!! warning

    Sometimes you need to modify your launch command which can be done with
    `SLURMCluster.job_class.submit_command`.    

```python
cluster.job_cls.submit_command = submit_command
cluster.job_cls.cancel_command = cancel_command
```