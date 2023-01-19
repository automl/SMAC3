__copyright__ = "Copyright 2023, automl.org"
__license__ = "3-clause BSD"

import numpy as np

import smac
from smac import Callback
from smac.runhistory import TrialInfo, TrialValue


class StoppingCallback(Callback):
    """Callback implementing the stopping criterion by Makarova et al. (2022) [1].

    [1] Makarova, Anastasia, et al. "Automatic Termination for Hyperparameter Optimization." First Conference on
    Automated Machine Learning (Main Track). 2022."""

    def __init__(self, beta=0.2, upper_bound_estimation_percentage=100, wait_iterations=20, n_points_lcb=1000):
        self.beta = beta
        self.upper_bound_estimation_percentage = upper_bound_estimation_percentage
        self.wait_iterations = wait_iterations
        self.n_points_lcb = n_points_lcb
        self.incumbent = None
        self.incumbent_value = None

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool:
        """Checks if the optimization should be stopped after the given trial."""

        print("End of tell")
        # todo - non-protected access to model
        model = smbo.intensifier.config_selector._model
        # for regret: sample ucb for all configurations in the runhistory
        # 1. get all configurations in the runhistory
        for trial_info, trial_value in smbo.runhistory.items():
            # convert config to vector

            config = smbo.runhistory.ids_config[trial_info.config_id].get_array()

            if self.incumbent is None:
                self.incumbent = trial_info
                self.incumbent_value = trial_value.cost
                self.incumbent_variance = trial_value.additional_info['']
            elif trial_value.cost < self.incumbent_value:
                self.incumbent = trial_info
                self.incumbent_value = trial_value.cost

            print(trial_value.additional_info)

            # todo - select x% of the configurations

        # 2. get model evaluations for all configurations
        # todo - what is y
        # todo - non-protected access to model etc.
        transformed_subset, y = smbo._intensifier._config_selector._runhistory_encoder.transform()

        # todo - dont rely on rf being used
        if model._rf is not None:
            mean, var = model.predict_marginalized(transformed_subset)
            # todo the predicted mean is lower than 0 - why?
            std = np.sqrt(var)
            ucbs = mean + np.sqrt(self.beta) * std
            print("mean", mean)
            print("std", std)
            print("ucbs", ucbs)
            min_ucb = np.min(ucbs)
            print("min_ucb", min_ucb)

        else:
            print("no model built yet")
        # for regret: sample lcb for a number of points
        # get inspired by random search acquisition function maximizer

        #regret = min_ucb - min_lcb

        # todo get trial info of incumbent
        # trial_info = ??

        #error = 0
        return True
