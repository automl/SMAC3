import numpy as np

from sklearn.svm import SVC
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario


X_train, y_train = np.random.randint(2, size=(20, 2)), np.random.randint(2, size=20)
X_val, y_val = np.random.randint(2, size=(5, 2)), np.random.randint(2, size=5)


def train_svc(config):
    model = SVC(C=config["C"])
    model.fit(X_train, y_train)

    return 1 - model.score(X_val, y_val)


if __name__ == "__main__":
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("C", 0, 1))

    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 10,  # Max number of function evaluations (the more the better)
        "cs": configspace,
    })

    smac = SMAC4BB(scenario=scenario, tae_runner=train_svc)
    best_found_config = smac.optimize()
