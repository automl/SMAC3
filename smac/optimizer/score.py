from typing import Tuple, Optional, List, Tuple

from ConfigSpace import Configuration, ConfigurationSpace
import numpy as np
from scipy.stats import spearmanr
# TODO replace by group shuffle split to take instances into account?
from sklearn.model_selection import ShuffleSplit

from smac.epm.base_epm import AbstractEPM
from smac.epm.random_epm import RandomEPM
from smac.optimizer.acquisition import AbstractAcquisitionFunction, EI, LogEI, LCB, PI
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM


class LossFunction:
    """Abstract loss function for internal model and acquisition function maximization.

    Paramaters
    ----------
    X_train : np.ndarray

    y_train : np.ndarray

    X_test : np.ndarray

    y_test : np.ndarray

    acq : AbstractAcquisitionFunction

    model : AbstractEPM

    config_space : ConfigurationSpace

    Returns
    -------
    float
    """

    def __call__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        acq: AbstractAcquisitionFunction,
        model: AbstractEPM,
        config_space: ConfigurationSpace,
    ) -> float:

        raise NotImplementedError()


class SpearmanLossFunction:
    """Computes the spearman rank coefficient between the test data and the acquisition function values.

    The returned loss is the negative spearman rank coefficient.

    Paramaters
    ----------
    X_train : np.ndarray

    y_train : np.ndarray

    X_test : np.ndarray

    y_test : np.ndarray

    acq : AbstractAcquisitionFunction

    model : AbstractEPM

    config_space : ConfigurationSpace

    Returns
    -------
    float
    """

    def __call__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        acq: AbstractAcquisitionFunction,
        model: AbstractEPM,
        config_space: ConfigurationSpace,
    ):

        model.train(X_train, y_train)
        acq.update(model=model, eta=np.min(y_train), num_data=len(X_train))
        configs = [Configuration(config_space, vector=x) for x in X_test]
        acquivals = acq(configurations=configs).flatten()
        return 1 - spearmanr(y_test, -acquivals)[0]


class TwoStepLookbackBOLossFunction:
    """Perform two steps of Bayesian optimization on the test data.

    The returned loss is the minimum of the two points selected by Bayesian optimization.

    Paramaters
    ----------
    X_train : np.ndarray

    y_train : np.ndarray

    X_test : np.ndarray

    y_test : np.ndarray

    acq : AbstractAcquisitionFunction

    model : AbstractEPM

    config_space : ConfigurationSpace

    Returns
    -------
    float
    """

    def two_step_lookback_bo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        acq: AbstractAcquisitionFunction,
        model: AbstractEPM,
        config_space: ConfigurationSpace,
    ):

        losses_for_split = []

        # First iteration of Bayesian optimization
        model.train(X_train, y_train)
        acq.update(model=model, eta=np.min(y_train), num_data=len(X_train))
        configs = [Configuration(config_space, vector=x) for x in X_test]
        acquivals = acq(configurations=configs).flatten()

        # Query test data and move queried data point to training data
        argmax = np.nanargmax(acquivals)
        x = X_test[argmax]
        y_ = y_test[argmax]
        losses_for_split.append(y_)
        X_test = np.delete(X_test, argmax, axis=0)
        y_test = np.delete(y_test, argmax, axis=0)
        X_train = np.concatenate((X_train, x.reshape((1, -1))), axis=0)
        y_train = np.concatenate((y_train, y_.reshape((1, 1))), axis=0)

        # Second iteration of Bayesian optimization
        # TODO only update the model here (important for a GP to only sample the hyperparameters once)
        model.train(X_train, y_train)
        acq.update(model=model, eta=np.min(y_train), num_data=len(X_train))
        configs = [Configuration(config_space, vector=x) for x in X_test]
        acquivals = acq(configurations=configs).flatten()

        # Query test data
        argmax = np.nanargmax(acquivals)
        y_ = y_test[argmax]

        losses_for_split.append(y_)
        return np.min(losses_for_split)


def adaptive_component_selection(
    runhistory: RunHistory,
    runhistory2EPM: AbstractRunHistory2EPM,
    default_model: AbstractEPM,
    default_acquisition_function: AbstractAcquisitionFunction,
    combinations: List[Tuple[AbstractEPM, AbstractAcquisitionFunction]],
    loss_function: Optional[LossFunction] = None,
    sampling_based: bool = True,
    min_test_size: int = 5,
    test_fraction: float = 0.2,
    min_runhistory_length: int = 10,
    n_splits: int = 200,
) -> Tuple[AbstractEPM, AbstractAcquisitionFunction]:
    """Select a model and an acquisition function given the runhistory data.

    Parameters
    ----------
    runhistory : RunHistory

    runhistory2EPM : AbstractRunHistory2EPM

    default_model : AbstractEPM

    default_acquisition_function : AbstractAcquisitionFunction

    combinations : list

    loss_function : LossFunction
        Uses the ``SpearmanLossFunction`` if no loss function is given.
    sampling_based : bool

    min_test_size : int

    test_fraction : float

    min_runhistory_length : int

    n_splits : int

    Returns
    -------
    AbstractEPM

    AbstractAcquisitionFunction
    """

    if len(runhistory.data) < min_runhistory_length:
        return default_model, default_acquisition_function

    if loss_function is None:
        loss_function = SpearmanLossFunction()

    X, y = runhistory2EPM.transform(runhistory)
    test_size = max(min_test_size, int(len(X) * test_fraction))

    combination_losses = []
    for model, acq in combinations:
        losses = []
        splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=1)
        for train_indices, test_indices in splitter.split(X, y):
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]

            loss = loss_function(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                acq=acq, model=model, config_space=runhistory2EPM.scenario.cs,
            )
            losses.append(loss)

        if sampling_based:
            combination_losses.append(losses)
        else:
            combination_losses.append(np.mean(losses))

    if sampling_based:
        wins = np.zeros((len(combinations), ))
        for cl in np.array(combination_losses).transpose():
            try:
                wins[np.nanargmin(cl)] += 1
            except ValueError:
                continue
        wins = wins / np.sum(wins)
        choice = np.random.choice(len(combinations), p=wins)
    else:
        choice = np.nanargmin(combination_losses)

    return combinations[choice][0], combinations[choice][1]
