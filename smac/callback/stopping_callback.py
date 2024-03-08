from abc import abstractmethod

import numpy as np

from smac.acquisition.function import LCB, UCB
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.main.smbo import SMBO
from smac.callback import Callback
from smac.runhistory import TrialInfo, TrialValue, TrialKey
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2023, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


def estimate_crossvalidation_statistical_error(std, folds, data_points_test, data_points_train):
    """Estimates the statistical error of a k-fold cross-validation according to [0].

    [0] Nadeau, Claude, and Yoshua Bengio. "Inference for the generalization error." Advances in neural information
    processing systems 12 (1999).

    Parameters
    ----------
    std : float
        Standard deviation of the cross-validation.
    folds : int
        Number of folds.
    data_points_test : int
        Number of data points in the test set.
    data_points_train : int
        Number of data points in the training set.

    Returns
    -------
    float
        Estimated statistical error.
    """
    return np.sqrt((1 / folds + data_points_test / data_points_train) * pow(std, 2))


class AbstractStoppingCallbackCallback:
    """Abstract class for stopping criterion callbacks."""

    @abstractmethod
    def log(self, smbo: SMBO, min_ubc: float, min_lcb: float, regret: float, statistical_error: float, triggered: bool)\
            -> None:
        """Logs the stopping criterion values.

        Parameters
        ----------
        smbo : SMBO
            The SMBO instance.
        min_ubc : float
            Minimum upper confidence bound.
        min_lcb : float
            Minimum lower confidence bound.
        regret : float
            Regret.
        statistical_error : float
            Statistical error.
        triggered : bool
            Whether the stopping criterion was triggered.
        """
        raise NotImplementedError()


class StoppingCallback(Callback):
    """Callback implementing the stopping criterion by Makarova et al. (2022) [0].

    [0] Makarova, Anastasia, et al. "Automatic Termination for Hyperparameter Optimization." First Conference on
    Automated Machine Learning (Main Track). 2022."""

    def __init__(self,
                 initial_beta=0.1,
                 update_beta=True,
                 upper_bound_estimation_rate=0.5,
                 wait_iterations=20,
                 n_points_lcb=1000,
                 model_log_transform=True,
                 statistical_error_threshold=None,
                 statistical_error_field_name='statistical_error',
                 do_not_trigger=False,
                 callbacks: list[AbstractStoppingCallbackCallback] = None):
        super().__init__()
        self._upper_bound_estimation_rate = upper_bound_estimation_rate
        self._wait_iterations = wait_iterations
        self._n_points_lcb = n_points_lcb
        self._model_log_transform = model_log_transform
        self._statistical_error_threshold = statistical_error_threshold
        self._statistical_error_field_name = statistical_error_field_name
        self._do_not_trigger = do_not_trigger
        self._callbacks = callbacks if callbacks is not None else []

        self._lcb = LCB(beta=initial_beta, update_beta=update_beta, beta_scaling_srinivas=True)
        self._ucb = UCB(beta=initial_beta, update_beta=update_beta, beta_scaling_srinivas=True)

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool:
        """Checks if the optimization should be stopped after the given trial."""

        # do not trigger stopping criterion before wait_iterations
        if smbo.runhistory.submitted < self._wait_iterations:
            return True

        # get statistical error of incumbent
        incumbent_config = smbo.intensifier.get_incumbent()
        trial_info_list = smbo.runhistory.get_trials(incumbent_config)

        if trial_info_list is None or len(trial_info_list) == 0:
            logger.warn("No trial info for incumbent found. Stopping criterion will not be triggered.")
            return True
        elif len(trial_info_list) > 1:
            raise ValueError("Currently, only one trial per config is supported.")

        trial_info = trial_info_list[0]

        trial_value = smbo.runhistory[TrialKey(config_id=smbo.runhistory.get_config_id(trial_info.config),
                                               instance=trial_info.instance, seed=trial_info.seed,
                                               budget=trial_info.budget)]

        if self._statistical_error_threshold is not None:
            incumbent_statistical_error = self._statistical_error_threshold
        else:
            incumbent_statistical_error = trial_value.additional_info[self._statistical_error_field_name]

        # compute regret
        model = smbo.intensifier.config_selector.model
        if model.fitted:
            configs = smbo.runhistory.get_configs(sort_by='cost')

            # update acquisition functions
            num_data = len(configs)
            self._lcb.update(model=model, num_data=num_data)
            self._ucb.update(model=model, num_data=num_data)

            # get pessimistic estimate of incumbent performance
            configs = configs[:int(self._upper_bound_estimation_rate * num_data)]
            min_ucb = min(-1 * self._ucb(configs))[0]
            if self._model_log_transform:
                min_ucb = np.exp(min_ucb)

            # get optimistic estimate of the best possible performance (min lcb of all configs)
            maximizer = LocalAndSortedRandomSearch(configspace=smbo.scenario.configspace,
                                                   acquisition_function=self._lcb,
                                                   challengers=1)
            # it is maximizing the negative lcb, thus, the minimum is found
            challenger_list = maximizer.maximize(previous_configs=[], n_points=self._n_points_lcb)
            min_lcb = -1 * self._lcb(challenger_list)[0]
            if self._model_log_transform:
                min_lcb = np.exp(min_lcb)[0]

            # compute regret
            regret = min_ucb - min_lcb

            # print stats
            logger.debug(f'Minimum UCB: {min_ucb}, minimum LCB: {min_lcb}, regret: {regret}, '
                         f'statistical error: {incumbent_statistical_error}')

            # we are stopping once regret < incumbent statistical error (return false = do not continue optimization
            continue_optimization = regret >= incumbent_statistical_error

            for callback in self._callbacks:
                callback.log(smbo, min_ucb, min_lcb, regret, incumbent_statistical_error, not continue_optimization)

            info_str = f'triggered after {len(smbo.runhistory)} evaluations with regret ' \
                       f'~{round(regret, 3)} and incumbent error ~{round(incumbent_statistical_error, 3)}.'
            if not continue_optimization:
                logger.info(f'Stopping criterion {info_str}')
            else:
                logger.debug(f'Stopping criterion not {info_str}')

            if self._do_not_trigger:
                return True
            else:
                return continue_optimization

        else:
            logger.debug("Stopping criterion not triggered as model is not built yet.")
            return True

    def __str__(self):
        return "StoppingCallback"
