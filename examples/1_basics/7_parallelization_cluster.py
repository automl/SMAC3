"""
Parallelization-on-Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of applying SMAC to optimize Branin using parallelization via Dask client on a 
SLURM cluster. If you do not want to use a cluster but your local machine, set dask_client
to `None` and pass `n_workers` to the `Scenario`.

:warning: On some clusters you cannot spawn new jobs when running a SLURMCluster inside a
job instead of on the login node. No obvious errors might be raised but it can hang silently.

Sometimes you need to modify your launch command which can be done with
`SLURMCluster.job_class.submit_command`. 

```python
cluster.job_cls.submit_command = submit_command
cluster.job_cls.cancel_command = cancel_command
```

Here we optimize the synthetic 2d function Branin.
We use the black-box facade because it is designed for black-box function optimization.
The black-box facade uses a :term:`Gaussian Process<GP>` as its surrogate model.
The facade works best on a numerical hyperparameter configuration space and should not
be applied to problems with large evaluation budgets (up to 1000 evaluations).
"""

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from smac import BlackBoxFacade, Scenario

__copyright__ = "Copyright 2023, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class Branin(object):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x0 = Float("x0", (-5, 10), default=-5, log=False)
        x1 = Float("x1", (0, 15), default=2, log=False)
        cs.add_hyperparameters([x0, x1])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Branin function

        Parameters
        ----------
        config : Configuration
            Contains two continuous hyperparameters, x0 and x1
        seed : int, optional
            Not used, by default 0

        Returns
        -------
        float
            Branin function value
        """
        x0 = config["x0"]
        x1 = config["x1"]
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)
        ret = a * (x1 - b * x0**2 + c * x0 - r) ** 2 + s * (1 - t) * np.cos(x0) + s

        return ret


if __name__ == "__main__":
    model = Branin()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, deterministic=True, n_trials=100)

    # Create cluster
    n_workers = 4  # Use 4 workers on the cluster
    # Please note that the number of workers is directly set in the
    # cluster / client. `scenario.n_workers` is ignored in this case.

    cluster = SLURMCluster(
        # This is the partition of our slurm cluster.
        queue="cpu_short",
        # Your account name
        # account="myaccount",
        cores=1,
        memory="1 GB",
        # Walltime limit for each worker. Ensure that your function evaluations
        # do not exceed this limit.
        # More tips on this here: https://jobqueue.dask.org/en/latest/advanced-tips-and-tricks.html#how-to-handle-job-queueing-system-walltime-killing-workers
        walltime="00:10:00",
        processes=1,
        log_directory="tmp/smac_dask_slurm",
    )
    cluster.scale(jobs=n_workers)

    # Dask will create n_workers jobs on the cluster which stay open.
    # Then, SMAC/Dask will schedule individual runs on the workers like on your local machine.
    client = Client(
        address=cluster,
    )
    # Instead, you can also do
    # client = cluster.get_client()

    # Now we use SMAC to find the best hyperparameters
    smac = BlackBoxFacade(
        scenario,
        model.train,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        dask_client=client,
    )

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
