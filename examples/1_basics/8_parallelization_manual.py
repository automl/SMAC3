"""
Parallelization by manually creating a Dask cluster, client and workers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of applying SMAC to optimize Branin using parallelization via Dask and
manually starting the dask scheduler and workers via the command line.

Use this example only if the more automated procedure in :ref:`Parallelization-on-Cluster`
is not suitable for you.

To run SMAC3 distributed we need to set up the following three components:

1. SMAC3 and a dask client. This will manage all workload, find new configurations 
   to evaluate and submit jobs via a dask client. As this runs Bayesian optimization 
   it should be executed on its own CPU.
2. The dask workers. They will do the actual work of running machine learning 
   algorithms and require their own CPU each.
3. The scheduler. It manages the communication between the dask client and the 
   different dask workers. As the client and all workers connect to the scheduler 
   it must be started first. This is a light-weight job and does not require its 
   own CPU.

We will now start these three components in reverse order: scheduler, workers 
and client. Also, in a real setup, the scheduler and the workers should be 
started from the command line and not from within a Python file via the 
subprocess module as done here (for the sake of having a self-contained example).
"""

import multiprocessing
import subprocess
import time

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float
from dask.distributed import Client

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


###########################################################################
# 0. Setup client-scheduler communication and temporary variables
# ===============================================================
#
# In this examples the dask scheduler is started without an explicit
# address and port. Instead, the scheduler takes a free port and stores
# relevant information in a file for which we provided the name and
# location. This filename is also given to the worker so they can find all
# relevant information to connect to the scheduler.

scheduler_file_name = "scheduler-file.json"
worker_processes = []

############################################################################
# 1. Start scheduler
# ==================
#
# Starting the scheduler is done with the following bash command:
#
# .. code:: bash
#
#     dask-scheduler --scheduler-file scheduler-file.json --idle-timeout 10
#
# We will now execute this bash command from within Python to have a
# self-contained example:


def cli_start_scheduler(scheduler_file_name):
    command = f"dask-scheduler --scheduler-file {scheduler_file_name} --idle-timeout 10"
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        check=True,
    )
    while proc.returncode is None:
        time.sleep(1)
    if proc.returncode != 0:
        raise ValueError(
            f"Scheduler failed to start with exit code {proc.returncode}. "
            f"Stdout: {proc.stdout}. Stderr: {proc.stderr}."
        )


if __name__ == "__main__":
    process_python_worker = multiprocessing.Process(
        target=cli_start_scheduler,
        args=(scheduler_file_name,),
    )
    process_python_worker.start()
    worker_processes.append(process_python_worker)

    # Wait a second for the scheduler to become available
    time.sleep(1)


############################################################################
# 2. Start two workers
# ====================
#
# Starting the scheduler is done with the following bash command:
#
# .. code:: bash
#
#     dask-worker --nthreads 1 --lifetime 35 --memory-limit 0 \
#         --scheduler-file scheduler-file.json
#
# We will now execute this bash command from within Python to have a
# self-contained example. # We disable dask's memory management by 
# passing ``--memory-limit`` as SMAC can do memory management itself.


def cli_start_worker(scheduler_file_name):
    command = (
        "dask-worker --nthreads 1 --lifetime 35 --memory-limit 0 "
        f"--scheduler-file {scheduler_file_name}"
    )
    proc = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    )
    while proc.returncode is None:
        time.sleep(1)
    if proc.returncode != 0:
        raise ValueError(
            f"Worker failed to start with exit code {proc.returncode}. "
            f"Stdout: {proc.stdout}. Stderr: {proc.stderr}."
        )


if __name__ == "__main__":
    for _ in range(2):
        process_cli_worker = multiprocessing.Process(
            target=cli_start_worker,
            args=(scheduler_file_name,),
        )
        process_cli_worker.start()
        worker_processes.append(process_cli_worker)

    # Wait a second for workers to become available
    time.sleep(1)


############################################################################
# 3. Creating a client in Python
# ==============================
#
# Finally we create a dask cluster which also connects to the scheduler via
# the information in the file created by the scheduler.

client = Client(scheduler_file=scheduler_file_name)


############################################################################
# Start SMAC
# ~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
    model = Branin()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, deterministic=True, walltime_limit=30)

    # Now we use SMAC to find the best hyperparameters
    smac = BlackBoxFacade(
        scenario,
        model.train,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        dask_client=client,
    )

    incumbent = smac.optimize()

    ############################################################################
    # Wait until all workers are closed
    # =================================
    #
    # This is only necessary if the workers are started from within this python
    # script. In a real application one would start them directly from the command
    # line.
    process_python_worker.join()
    for process in worker_processes:
        process.join()
