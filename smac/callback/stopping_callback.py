__copyright__ = "Copyright 2023, automl.org"
__license__ = "3-clause BSD"

import numpy as np

from smac.acquisition.function import LCB, UCB
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.main.smbo import SMBO
from smac.callback import Callback
from smac.runhistory import TrialInfo, TrialValue, TrialKey


class StoppingCallback(Callback):
    """Callback implementing the stopping criterion by Makarova et al. (2022) [1].

    [1] Makarova, Anastasia, et al. "Automatic Termination for Hyperparameter Optimization." First Conference on
    Automated Machine Learning (Main Track). 2022."""

    def __init__(self, initial_beta=0.2, upper_bound_estimation_percentage=1, wait_iterations=20, n_points_lcb=1000):
        self.beta = initial_beta
        self.upper_bound_estimation_percentage = upper_bound_estimation_percentage
        self.wait_iterations = wait_iterations
        self.n_points_lcb = n_points_lcb

        # todo - add static beta version
        self.lcb = LCB(beta=self.beta)
        self.ucb = UCB(beta=self.beta)

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool:
        """Checks if the optimization should be stopped after the given trial."""

        # do not trigger stopping criterion before wait_iterations
        if smbo.runhistory.submitted < self.wait_iterations:
            return True

        # update statistical error of incumbent
        # todo - only recompute if incumbent changed (maybe easier one additional info has the statistical error)
        incumbent_config = smbo.intensifier.get_incumbent()
        trial_infos = smbo.runhistory.get_trials(incumbent_config)
        trial_values = [smbo.runhistory[TrialKey(config_id=smbo.runhistory.get_config_id(trial_info.config),
                                                 instance=trial_info.instance, seed=trial_info.seed,
                                                 budget=trial_info.budget)] for trial_info in trial_infos]

        if len(trial_values) != 1:
            raise ValueError("Currently, only one trial per config is supported.")
        trial_value = trial_values[0]

        # todo - don't assume train/test split sizes - rather add 'statistical error' to additional info
        std_incumbent = trial_value.additional_info['std_crossval']
        folds = trial_value.additional_info['folds']
        data_points = trial_value.additional_info['data_points']
        data_points_test = data_points / folds
        data_points_train = data_points - data_points_test
        incumbent_statistical_error = (1/folds + data_points_test / data_points_train) * pow(std_incumbent, 2)

        # compute regret
        # todo - dont rely on rf being used
        model = smbo.intensifier.config_selector._model
        if model._rf is not None:
            # update acquisition functions
            num_data = len(smbo.intensifier.config_selector._get_evaluated_configs())
            self.lcb.update(model=model, num_data=num_data)
            self.ucb.update(model=model, num_data=num_data)

            # get pessimistic estimate of incumbent performance
            configs = smbo.runhistory.get_configs(sort_by='cost')
            print([(config, self.ucb([config])) for config in configs])
            configs = configs[:int(self.upper_bound_estimation_percentage * len(configs))]
            min_ucb = min(self.ucb(configs))
            print("min ucb", min_ucb)

            # get optimistic estimate of the best possible performance (min lcb of all configs)
            maximizer = LocalAndSortedRandomSearch(configspace=smbo._scenario.configspace,
                                                   acquisition_function=self.lcb,
                                                   challengers=1)
            # it is maximizing the negative lcb, thus, the minimum is found
            challenger_list = maximizer.maximize(previous_configs=[], n_points=self.n_points_lcb)
            min_lcb = self.lcb(challenger_list)[0]
            print("min lcb", min_lcb)

            # decide whether to stop
            regret = min_ucb - min_lcb

            print("regret", regret)
            print("statistical error", incumbent_statistical_error)

            # we are stopping once regret < incumbent statistical error (return false = do not continue)
            return regret >= incumbent_statistical_error

        else:
            print("no model built yet")

        return True
