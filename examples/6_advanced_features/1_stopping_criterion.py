"""
Use Stopping Criterion to stop the Optimization Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to use the stopping criterion by Makarova et al. (2021) [1] to stop the optimization process. It
is implemented in SMAC as a callback function.

[1] Makarova, Anastasia, et al. "Automatic Termination for Hyperparameter Optimization." First Conference on
    Automated Machine Learning (Main Track). 2022.
"""

from typing import Any, Union

from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from smac.callback import StoppingCallback
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from smac.callback.stopping_callback import estimate_crossvalidation_statistical_error
from smac.intensifier import Intensifier

statistical_error_field_name = 'statistical_error'
cv_folds = 5

iris = datasets.load_iris()
data_points_test = len(iris.data) // cv_folds
data_points_train = len(iris.data) - data_points_test


def train(config: Configuration, seed: int = 0) -> tuple[Any, dict[str, Union[int, Any]]]:
    classifier = SVC(C=config["C"], random_state=seed)
    scores = cross_val_score(classifier, iris.data, iris.target, cv=cv_folds)

    estimate = estimate_crossvalidation_statistical_error(
        std=np.std(scores),
        folds=cv_folds,
        data_points_test=data_points_test,
        data_points_train=data_points_train
    )

    # In addition to the mean score, we also return the statistical error for the stopping criterion
    return 1 - np.mean(scores), {statistical_error_field_name: estimate}


# Specify the configuration space, scenario
configspace = ConfigurationSpace({"C": (0.100, 1000.0)})
scenario = Scenario(configspace, deterministic=True, n_trials=50)

# Modify intensifier to only allow 1 evaluation per configuration
intensifier = Intensifier(scenario, max_config_calls=1)
# HPO Facade uses a random forest model with log_y=True, so we also need to transform the target values
model_log_transform = True
# Adjust to desired logging level
logging_level = 0

# Use SMAC to find the best configuration/hyperparameters
callback = StoppingCallback(
    wait_iterations=15,
    model_log_transform=model_log_transform,
    statistical_error_field_name=statistical_error_field_name
)

smac = HyperparameterOptimizationFacade(
    scenario,
    train,
    callbacks=[callback],
    logging_level=logging_level,
    intensifier=intensifier
)
incumbent = smac.optimize()

print(f'Result of Optimization: \n\n {incumbent}')
