from typing import Tuple, Callable

from ConfigSpace import Configuration
import numpy as np
from scipy.stats import spearmanr
# TODO replace by group shuffle split to take instances into account?
from sklearn.model_selection import ShuffleSplit

from smac.epm.base_epm import AbstractEPM
from smac.epm.random_epm import RandomEPM
from smac.optimizer.acquisition import AbstractAcquisitionFunction, EI, LogEI, LCB, PI
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM


def spearman(X, acq, model, runhistory2EPM, test_indices, train_indices, y):
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    model.train(X_train, y_train)
    acq.update(model=model, eta=np.min(y_train), num_data=len(X_train))
    configs = [Configuration(runhistory2EPM.scenario.cs, vector=x) for x in X_test]
    acquivals = acq(configurations=configs).flatten()
    return 1 - spearmanr(y_test, -acquivals)[0]


def two_step_lookback_bo(X, acq, model, runhistory2EPM, test_indices, train_indices, y):
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    losses_for_split = []
    model.train(X_train, y_train)
    acq.update(model=model, eta=np.min(y_train), num_data=len(X_train))
    configs = [Configuration(runhistory2EPM.scenario.cs, vector=x) for x in X_test]
    acquivals = acq(configurations=configs).flatten()
    argmax = np.nanargmax(acquivals)
    x = X_test[argmax]
    y_ = y_test[argmax]
    losses_for_split.append(y_)
    X_test = np.delete(X_test, argmax, axis=0)
    y_test = np.delete(y_test, argmax, axis=0)
    X_train = np.concatenate((X_train, x.reshape((1, -1))), axis=0)
    y_train = np.concatenate((y_train, y_.reshape((1, 1))), axis=0)
    model.train(X_train, y_train)
    acq.update(model=model, eta=np.min(y_train), num_data=len(X_train))
    configs = [Configuration(runhistory2EPM.scenario.cs, vector=x) for x in X_test]
    acquivals = acq(configurations=configs).flatten()
    argmax = np.nanargmax(acquivals)
    y_ = y_test[argmax]
    losses_for_split.append(y_)
    l = np.min(losses_for_split)
    return l


def great_new_selection_function(
    runhistory: RunHistory,
    runhistory2EPM: AbstractRunHistory2EPM,
    default_model: AbstractEPM,
    default_acquisition_function: AbstractAcquisitionFunction,
    loss_function: Callable = spearman,
    sampling_based: bool = True,
) -> Tuple[AbstractEPM, AbstractAcquisitionFunction]:

    if len(runhistory.data) < 10:
        return default_model, default_acquisition_function

    X, y = runhistory2EPM.transform(runhistory)
    test_size = max(5, int(len(X) * 0.2))

    ei = LogEI(par=0.0, model=default_model)
    ei2 = LogEI(par=0.1, model=default_model)
    ei3 = LogEI(par=1.0, model=default_model)
    ei4 = LogEI(par=-0.1, model=default_model)
    ei5 = LogEI(par=-1, model=default_model)
    lcb = LCB(model=default_model)
    pi = PI(model=default_model)

    combinations = [
        (default_model, ei),
        (default_model, ei2),
        (default_model, ei3),
        (default_model, ei4),
        (default_model, ei5),
        (RandomEPM(np.random.RandomState(1), types=default_model.types.copy(), bounds=default_model.bounds), ei),
        (default_model, lcb),
        (default_model, pi),
    ]

    combination_losses = []
    for model, acq in combinations:
        losses = []
        splitter = ShuffleSplit(n_splits=200, test_size=test_size, random_state=1)
        for train_indices, test_indices in splitter.split(X, y):
            loss = loss_function(X, acq, model, runhistory2EPM, test_indices, train_indices, y)
            losses.append(loss)

        if sampling_based:
            combination_losses.append(losses)
        else:
            combination_losses.append(np.mean(losses))

    if sampling_based:
        wins = np.zeros((len(combinations), ))
        for cl in np.array(combination_losses).transpose():
            wins[np.argmin(cl)] += 1
        wins = wins / np.sum(wins)
        choice = np.random.choice(len(combinations), p=wins)
        print(wins, X.shape, test_size, choice, len(combinations))
    else:
        choice = np.nanargmin(combination_losses)
        print(combination_losses, X.shape, test_size, choice)
    return combinations[choice][0], combinations[choice][1]
