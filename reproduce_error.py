from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace import Normal, Integer, Float, Beta, Categorical

from smac import Scenario, HyperparameterOptimizationFacade
from multiprocessing import freeze_support
from itertools import product

import numpy as np
import time


cs = ConfigurationSpace(
    space={
        "lr": Float(
            'lr',
            bounds=(1e-5, 1e-1),
            default=1e-3,
            log=True,
            distribution=Normal(1e-3, 1e-1)
        ),
        "dropout": Float(
            'dropout',
            bounds=(0, 0.99),
            default=0.25,
            distribution=Beta(alpha=2, beta=4)
        ),
        "activation": Categorical(
            'activation',
            items=['tanh', 'relu'],
            weights=[0.2, 0.8]
        ),
    },
    seed=1234,
)


def get_fake_performance(config : Configuration, instance: str, seed: int = 0):
    return [np.random.uniform()]

def run_smac(fids):
    inst_feats = {str(arg): [idx] for idx, arg in enumerate(fids)}
    scenario = Scenario(
        cs,
        name=str(int(time.time())) + "-" + "example",
        deterministic=False,
        min_budget=2,
        max_budget=20,
        n_trials=200,
        instances=fids,
        instance_features=inst_feats,
        output_directory="smac3_output", 
        n_workers=2
    )
    smac = HyperparameterOptimizationFacade(scenario, get_fake_performance)
    smac.optimize()

if __name__ == "__main__":
    freeze_support()
    run_smac([1,2,3])
