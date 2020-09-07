import warnings

import numpy as np

from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier


# A common function to be optimized by a Real valued Intensifier
digits = load_digits()


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
def mlp_from_cfg(cfg, seed, instance, budget, **kwargs):
    """
        Creates a MLP classifier from sklearn and fits the given data on it.
        This is the function-call we try to optimize. Chosen values are stored in
        the configuration (cfg).

        Parameters
        ----------
        cfg: Configuration
            configuration chosen by smac
        seed: int or RandomState
            used to initialize the rf's random generator
        instance: str
            used to represent the instance to use (just a placeholder for this example)
        budget: float
            used to set max iterations for the MLP

        Returns
        -------
        float
    """

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        mlp = MLPClassifier(
            hidden_layer_sizes=[cfg["n_neurons"]] * cfg["n_layer"],
            batch_size=cfg['batch_size'],
            activation=cfg['activation'],
            learning_rate_init=cfg['learning_rate_init'],
            max_iter=int(np.ceil(budget)),
            random_state=seed)

        # returns the cross validation accuracy
        # to make CV splits consistent
        cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        score = cross_val_score(mlp, digits.data, digits.target, cv=cv, error_score='raise')

    return 1 - np.mean(score)  # Because minimize!
