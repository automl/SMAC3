from abc import abstractmethod
from typing import Optional, Union

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from smac import RunHistory
from smac.acquisition.function import LCB, UCB
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.callback import Callback
from smac.intensifier import SuccessiveHalving
from smac.main.smbo import SMBO
from smac.runhistory import TrialInfo, TrialKey, TrialValue
from smac.runhistory.encoder import AbstractRunHistoryEncoder
from smac.utils.logging import get_logger

logger = get_logger(__name__)


def estimate_crossvalidation_statistical_error(
    std: float, folds: int, data_points_test: int, data_points_train: int
) -> float:
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
    def log(
        self,
        min_ucb: float,
        min_lcb: float,
        regret: float,
        statistical_error: float,
        triggered: bool,
        **kwargs: dict,
    ) -> None:
        """Logs the stopping criterion values.

        Parameters
        ----------
        min_ucb : float
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
    """
    Callback implementing the stopping criterion by Makarova et al. (2022) [0].

    [0] Makarova, Anastasia, et al. "Automatic Termination for Hyperparameter Optimization." First Conference on
    Automated Machine Learning (Main Track). 2022.
    """

    def __init__(
        self,
        initial_beta: float = 0.1,
        update_beta: bool = True,
        upper_bound_estimation_rate: float = 0.5,
        wait_iterations: int = 20,
        n_points_lcb: int = 1000,
        statistical_error_threshold: Union[float, None] = None,
        statistical_error_field_name: str = "statistical_error",
        do_not_trigger: bool = False,
        highest_fidelity_only: bool = True,
        callbacks: list[AbstractStoppingCallbackCallback] = None,
    ):
        super().__init__()
        self._upper_bound_estimation_rate = upper_bound_estimation_rate
        self._wait_iterations = wait_iterations
        self._n_points_lcb = n_points_lcb
        self._statistical_error_threshold = statistical_error_threshold
        self._statistical_error_field_name = statistical_error_field_name
        self._do_not_trigger = do_not_trigger
        self._highest_fidelity_only = highest_fidelity_only
        self._callbacks = callbacks if callbacks is not None else []

        self._lcb = LCB(beta=initial_beta, update_beta=update_beta, beta_scaling_srinivas=True)
        self._ucb = UCB(beta=initial_beta, update_beta=update_beta, beta_scaling_srinivas=True)

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool:
        """Checks if the optimization should be stopped after the given trial."""
        # do not trigger stopping criterion before wait_iterations
        if smbo.runhistory.submitted < self._wait_iterations:
            return True

        max_budget: float = smbo.intensifier._max_budget  # type: ignore[attr-defined]

        # in the case of the highest fidelity only, check if the received config is on the highest fidelity and
        # if the model is trained on the highest fidelity
        if self._highest_fidelity_only:
            assert isinstance(smbo.intensifier, SuccessiveHalving)
            intensifier: SuccessiveHalving = smbo.intensifier
            if (
                info.budget != intensifier._max_budget
                or intensifier.config_selector._model_trained_on_budget != intensifier._max_budget
            ):
                return True

        # get statistical error of incumbent
        incumbent_config = smbo.intensifier.get_incumbent()
        # this gets the highest observed budget ONLY of the incumbent config
        trial_info_list = smbo.runhistory.get_trials(incumbent_config)

        if trial_info_list is None or len(trial_info_list) == 0:
            logger.warning("No trial info for incumbent found. Stopping criterion will not be triggered.")
            return True
        elif len(trial_info_list) > 1:
            raise ValueError("Currently, only one trial per config is supported.")

        trial_info = trial_info_list[0]

        # check if the incumbent is on the highest fidelity
        if self._highest_fidelity_only and trial_info.budget != intensifier._max_budget:
            return True

        trial_value = smbo.runhistory[
            TrialKey(
                config_id=smbo.runhistory.get_config_id(trial_info.config),
                instance=trial_info.instance,
                seed=trial_info.seed,
                budget=trial_info.budget,
            )
        ]

        if self._statistical_error_threshold is not None:
            incumbent_statistical_error = self._statistical_error_threshold
        else:
            incumbent_statistical_error = trial_value.additional_info[self._statistical_error_field_name]

        # compute regret
        assert smbo.intensifier.config_selector.model is not None
        model = smbo.intensifier.config_selector.model
        if model.fitted:
            if not self._highest_fidelity_only:
                configs = smbo.runhistory.get_configs(sort_by="cost")
            else:
                configs = self.get_configs_for_budget(smbo.runhistory, self._upper_bound_estimation_rate, max_budget)

            runhistory_encoder = smbo.intensifier.config_selector.runhistory_encoder

            # update acquisition functions
            num_data = len(configs)
            self._lcb.update(model=model, num_data=num_data)
            self._ucb.update(model=model, num_data=num_data)

            # get pessimistic estimate of incumbent performance
            configs = configs[: int(self._upper_bound_estimation_rate * num_data)]
            min_lcb, min_ucb = self.compute_min_lcb_ucb(
                self._ucb, self._lcb, self._n_points_lcb, configs, smbo.scenario.configspace, runhistory_encoder
            )

            # compute regret
            regret = min_ucb - min_lcb

            # print stats
            logger.debug(
                f"Minimum UCB: {min_ucb}, minimum LCB: {min_lcb}, regret: {regret}, "
                f"statistical error: {incumbent_statistical_error}"
            )

            # we are stopping once regret < incumbent statistical error (return false = do not continue optimization
            continue_optimization = regret >= incumbent_statistical_error

            for callback in self._callbacks:
                callback.log(min_ucb, min_lcb, regret, incumbent_statistical_error, not continue_optimization)

            info_str = (
                f"triggered after {len(smbo.runhistory)} evaluations with regret "
                f"~{round(regret, 3)} and incumbent error ~{round(incumbent_statistical_error, 3)}."
            )
            if not continue_optimization:
                logger.info(f"Stopping criterion {info_str}")
            else:
                logger.debug(f"Stopping criterion not {info_str}")

            if self._do_not_trigger:
                return True
            else:
                return continue_optimization

        else:
            logger.debug("Stopping criterion not triggered as model is not built yet.")
            return True

    @staticmethod
    def get_configs_for_budget(
        runhistory: RunHistory, upper_bound_estimation_rate: float, budget: float
    ) -> list[Configuration]:
        """
        Returns the configs for the given budget sorted by cost.

        Parameters
        ----------
        runhistory : RunHistory
            The runhistory from which the configs should be extracted.
        upper_bound_estimation_rate : float
            The rate of configs that should be considered for the upper bound estimation.
        budget : float
            The budget for which the configs should be extracted.

        Returns
        -------
        configs_ucb : list[Configuration]
        """
        configs_ucb = runhistory.get_configs_per_budget([budget])
        trials_ucb = []
        for config in configs_ucb:
            trials = runhistory.get_trials(config, highest_observed_budget_only=False)
            for trial in trials:
                if trial.budget == budget:
                    trial_value = runhistory[
                        TrialKey(
                            config_id=runhistory.get_config_id(config),
                            instance=trial.instance,
                            seed=trial.seed,
                            budget=trial.budget,
                        )
                    ]
                    trials_ucb.append((config, trial_value.cost))
        trials_ucb.sort(key=lambda trial_ucb: trial_ucb[1])
        amount_selected_configs = int(upper_bound_estimation_rate * len(trials_ucb))
        amount_selected_configs = max(1, amount_selected_configs)
        trials_ucb = trials_ucb[:amount_selected_configs]
        configs_ucb = [trial[0] for trial in trials_ucb]

        return configs_ucb

    @staticmethod
    def compute_min_lcb_ucb(
        ucb: UCB,
        lcb: LCB,
        n_points_lcb: int,
        configs: list[Configuration],
        configspace: ConfigurationSpace,
        runhistory_encoder: Optional[AbstractRunHistoryEncoder] = None,
    ) -> tuple[float, float]:
        """
        Computes the minimum lcb and ucb of the given configs.

        Parameters
        ----------
        ucb : UCB
            The ucb acquisition function.
        lcb : LCB
            The lcb acquisition function.
        n_points_lcb : int
            The number of points that should be sampled for the lcb.
        configs : list[Configuration]
            The configs for computing the ucb.
        configspace : ConfigurationSpace
            The configspace, needed for optimizing lcb.
        runhistory_encoder : Optional[AbstractRunHistoryEncoder]
            The runhistory encoder, needs to be given if the costs in the runhistory are encoded.
        """
        min_ucb = ucb(configs)
        min_ucb *= -1
        if runhistory_encoder is not None:
            min_ucb = runhistory_encoder.transform_response_values_inverse(min_ucb)
        min_ucb = min(min_ucb)[0]
        # get optimistic estimate of the best possible performance (min lcb of all configs)
        maximizer = LocalAndSortedRandomSearch(configspace=configspace, acquisition_function=lcb, challengers=1)
        # SMBO is maximizing the negative lcb, thus, we need to invert the lcb
        challenger_list = maximizer.maximize(previous_configs=[], n_points=n_points_lcb)

        min_lcb = lcb(challenger_list)
        min_lcb *= -1
        if runhistory_encoder is not None:
            min_lcb = runhistory_encoder.transform_response_values_inverse(min_lcb)
        min_lcb = min(min_lcb)[0]
        return min_lcb, min_ucb

    def __str__(self) -> str:
        return "StoppingCallback"
