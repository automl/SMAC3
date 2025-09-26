# Parallelism

To facilitate parallel execution, SMAC supports executing multiple workers simultaneously via [Dask](https://www.dask.org/). Using this functionality, splits SMAC into a main process, and DASK workers which handle the execution.
The main job handles the optimization process, and coordinates the executor jobs. The executors are queried with the target function and hyperparameter configurations, execute them, and return their result. The executors remain open between different executions.

!!! note
    
    Please keep in mind that additional workers are only used to evaluate trials. The main thread still orchestrates the
    optimization process, including training the surrogate model.

!!! warning

    When using multiple workers, SMAC is not reproducible anymore.


## Parallelizing Locally

To utilize parallelism locally, that means running workers on the same machine as the main jobs, specify the ``n_workers`` keyword when creating the scenario.
```python
Scenario(model.configspace, n_workers=5)
``` 


## Parallelizing on SLURM

To utilize this split of main and execution jobs on a [SLURM cluster](https://slurm.schedmd.com/), SMAC supports manually specifying a [Dask](https://www.dask.org/) client.
This allows executing the target function on dedicated SLURM jobs that are necessarily configured with the same hardware requirements,.

!!! note

    While most SLURM clusters behave similarly, the example DASK client might not work for every cluster. For example, some clusters only allow spawning new jobs 
    from the login node.

To configure SMAC properly for each cluster, you need to know the ports which allow communication between main and worker jobs. The dask client is then created as follows:

```python
...
from smac import BlackBoxFacade, Scenario
from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(
    queue="partition_name",                 # Name of the partition
    cores=4,                                # CPU cores requested
    memory="4 GB",                          # RAM requested
    walltime="00:10:00",                    # Walltime limit for a runner job. 
    processes=1,                            # Number of processes per worker
    log_directory="tmp/smac_dask_slurm",    # Logging directory
    nanny=False,                            # False unless you want to use pynisher
    worker_extra_args=[
        "--worker-port",                    # Worker port range 
        "60010:60100"],                     # Worker port range 
    scheduler_options={
        "port": 60001,                      # Main Job Port
    },
)
cluster.scale(jobs=n_workers)

# Dask creates n_workers jobs on the cluster which stay open.
client = Client(
    address=cluster,
)

#Dask waits for n_workers workers to be created
client.wait_for_workers(n_workers)

# Now we use SMAC to find the best hyperparameters
smac = BlackBoxFacade(
    scenario,                               # Pass scenario
    model.train,                            # Pass Pass target-function
    overwrite=True,                         # Overrides any previous result
    dask_client=client,                     # Pass dask_client
)
incumbent = smac.optimize()
```

The full example of this code is given in [parallelism example](../examples/1%20Basics/7_parallelization_cluster.md).