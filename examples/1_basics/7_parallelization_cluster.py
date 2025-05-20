"""Parallelization on Cluster

An example of applying SMAC to optimize Branin using parallelization via Dask client on a
SLURM cluster. If you do not want to use a cluster but your local machine, set dask_client
to `None` and pass `n_workers` to the `Scenario`.
"""

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from smac import BlackBoxFacade, Scenario

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from ConfigSpace import Configuration  # for type hints

class Branin:
    def __init__(self, seed: int = 0):
        cs = ConfigurationSpace(seed=seed)
        x0 = Float("x0", (-5, 10), default=-5, log=False)
        x1 = Float("x1", (0, 15), default=2, log=False)
        cs.add([x0, x1])

        self.cs = cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        x0 = config["x0"]
        x1 = config["x1"]
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)
        return a * (x1 - b * x0**2 + c * x0 - r) ** 2 \
             + s * (1 - t) * np.cos(x0) + s

if __name__ == "__main__":
    model = Branin()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(
        model.cs,
        deterministic=True,
        n_trials=100,
        trial_walltime_limit=100,
        n_workers=5,
    )

    n_workers = 5  # Use 5 workers on the cluster
    # Please note that the number of workers is directly set in the
    # cluster / client. `scenario.n_workers` is ignored in this case.

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
        # account="myaccount",                  # Account name on the cluster (optional)
    )
    cluster.scale(jobs=n_workers)

    # Dask will create n_workers jobs on the cluster which stay open.
    # Then, SMAC/Dask will schedule individual runs on the workers like on your local machine.
    client = Client(
        address=cluster,
    )
    client.wait_for_workers(n_workers)
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
    print(f"Best configuration found: {incumbent}")