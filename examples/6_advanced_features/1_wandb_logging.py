"""
Use Weights and Biases for logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to use Weights and Biases for logging.

"""
from __future__ import annotations

from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from smac import Callback
import smac
from wandb import Table


class WandBCallback(Callback):
    def __init__(
            self,
            project: str,
            entity: str,
            id: str | None = None,
            outdir: str | None = None,
            mode: str | None = None,
            resume: str = "allow",
            job_type: str | None = None,
            group: str | None = None,
            config: dict | str | None = None,
            save_code: bool = True,
            **kwargs
        ) -> None:
        import wandb
        self.run = wandb.init(
            id=id,
            resume=resume,
            mode=mode,
            project=project,
            job_type=job_type,
            entity=entity,
            group=group,
            dir=outdir,
            config=config,
            save_code=save_code,
            **kwargs
        )
        super().__init__()
    

    def on_end(self, smbo: smac.main.smbo.SMBO) -> None:
        intensifier_data = smbo.intensifier.get_data()
        trajectory = intensifier_data["trajectory"]
        import pandas as pd
        df = pd.DataFrame(data=trajectory)
        print(df)
        # trajectory = Table(dataframe=df, allow_mixed_types=True)
        df["costs"] = df["costs"].apply(lambda x: x[0])  # TODO properly log multi costs
        for index, row in df.iterrows():
            print(dict(row))
            self.run.log(dict(row))
        self.run.finish()
        return super().on_end(smbo)




iris = datasets.load_iris()


def train(config: Configuration, seed: int = 0) -> float:
    classifier = SVC(C=config["C"], random_state=seed)
    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores)


configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=True, n_trials=100)

wandb_callback = WandBCallback(
    project="smac-dev",
    entity="benjamc",
    config=Scenario.make_serializable(scenario),
)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train, callbacks=[wandb_callback], overwrite=True)
incumbent = smac.optimize()