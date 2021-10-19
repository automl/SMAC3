"""
Non-Deterministic Gradient-Boosting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We optimize a GradientBoosting on an artificially created binary classification dataset.
The results are not deterministic so we need to evaluate each configuration
multiple times. To ensure fair comparison, SMAC will only sample from a fixed set of random seeds and apply them to
control the randomness of the function to be evaluated.

To evaluate undeterministic functions, we need to set "deterministic" as "false".
Additional to the configuration, the function should make use of the seed parameter as well.
"""

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np

from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


# load data and split it into training and test dataset
X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:8400], X[8400:]
y_train, y_test = y[:8400], y[8400:]


# Gradient Boosting scored with cross validation
def xgboost_from_cfg(cfg, seed=0):
    # TODO: Do we need seed at all?
    clf = GradientBoostingClassifier(**cfg, random_state=0).fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train)

    return 1 - np.mean(scores)


if __name__ == "__main__":
    # creating a Configuration Space with every parameter over which SMAC is going to optimize
    cs = ConfigurationSpace()

    max_depth = UniformIntegerHyperparameter("max_depth", 1, 10, default_value=3)
    cs.add_hyperparameter(max_depth)

    learning_rate = UniformFloatHyperparameter("learning_rate", 0.01, 1.0, default_value=1.0, log=True)
    cs.add_hyperparameter(learning_rate)

    min_samples_split = UniformFloatHyperparameter("min_samples_split", 0.01, 1.0, default_value=0.1, log=True)
    max_features = UniformIntegerHyperparameter("max_features", 2, 10, default_value=4)
    cs.add_hyperparameters([min_samples_split, max_features])

    subsample = UniformFloatHyperparameter("subsample", 0.5, 1, default_value=0.8)
    cs.add_hyperparameter(subsample)

    print("Default cross validation score: %.2f" % (xgboost_from_cfg(cs.get_default_configuration())))
    cfg = cs.get_default_configuration()
    clf = GradientBoostingClassifier(**cfg, random_state=0).fit(X_train, y_train)
    def_test_score = 1 - clf.score(X_test, y_test)
    print("Default test score: %.2f" % def_test_score)

    # scenario object
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": 100,
        "cs": cs,
        # the evaluations are not deterministic, we need to repeat each
        # configuration several times and take the mean value of these repetitions
        "deterministic": "false",
        "wallclock_limit": 120,
        "maxR": 3,  # Each configuration will be evaluated maximal 3 times with various seeds
        "minR": 1,  # Each configuration will be repeated at least 1 time with different seeds
    })

    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(0), tae_runner=xgboost_from_cfg)

    # the optimization process is called
    incumbent = smac.optimize()

    # a classifier is trained with the hyperparameters returned from the optimizer
    clf_incumbent = GradientBoostingClassifier(**incumbent, random_state=0).fit(X_train, y_train)

    # evaluated on test
    inc_value_1 = 1 - clf_incumbent.score(X_test, y_test)
    print("Score on test set: %.2f" % (inc_value_1))
