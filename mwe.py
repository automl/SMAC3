#!/usr/bin/env python3

import logging
import time
from pathlib import Path
import random

from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import AlgorithmConfigurationFacade, Scenario
from dask_jobqueue import SLURMCluster

cs = ConfigurationSpace(seed=0)
cs.add_hyperparameters([
    Float("x", [0, 1], default=0.75),
])

def run_trial(config: Configuration, seed: int = 0) -> float:
    x = config["x"]
    path = Path(f"tmp/mwe/logs/log.{x}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"start with {x=}")
    return (x-0.5)**2


if __name__ == "__main__":
    cluster = SLURMCluster(
        cores=1,
        memory="4 GB",
        queue="cpu_short",
        # interface="eth0",
        walltime=f"00:10:00",
        # job_script_prologue=[
        #     "ulimit -c 0",
        # ]
        log_directory="tmp/mwe/slurm"
        
    )
    cluster.scale(jobs=10)

    scenario = Scenario(
        cs,
        deterministic=True,
        walltime_limit=600,
        n_trials=400,
        use_default_config=True,
        crash_cost = 2,
        trial_walltime_limit=1000,
        n_workers=2,
        
    )

    smac = AlgorithmConfigurationFacade(
        scenario,
        run_trial,
        overwrite=True,
        dask_client=None#cluster.get_client(),
    )
    smac.intensifier._retries = 10**6
    time.sleep(10)
    incumbent = smac.optimize()

    logging.info(f"Incumbent: {incumbent.get_dictionary()}")