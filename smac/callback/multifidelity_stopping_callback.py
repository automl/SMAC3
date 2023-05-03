from typing import Optional

import numpy as np

from smac import RunHistory, Scenario
from smac.acquisition.function import LCB, UCB
from smac.intensifier.stage_information import Stage
from smac.runhistory import TrialKey
from smac.runhistory.encoder import RunHistoryEncoder

__copyright__ = "Copyright 2023, automl.org"
__license__ = "3-clause BSD"


class MultiFidelityStoppingCallback:
    def __init__(
        self,
        initial_beta: float = 0.1,
        update_beta: bool = True,
        upper_bound_estimation_rate: float = 0.5,
        n_points_lcb: int = 1000,
        statistical_error_threshold: Optional[float] = None,
        statistical_error_field_name: Optional[str] = "statistical_error",
        statistical_error_estimation_only_incumbent: bool = True,
        statistical_error_config_estimation_percentage: Optional[float] = None,
        callbacks: list = None,
    ):
        super().__init__()
        self._upper_bound_estimation_rate = upper_bound_estimation_rate
        self._n_points_lcb = n_points_lcb
        self._statistical_error_threshold = statistical_error_threshold
        self._statistical_error_field_name = statistical_error_field_name
        self._statistical_error_estimation_only_incumbent = statistical_error_estimation_only_incumbent
        self._statistical_error_config_estimation_percentage = statistical_error_config_estimation_percentage
        self._callbacks = callbacks if callbacks is not None else []

        self._lcb = LCB(beta=initial_beta, update_beta=update_beta, beta_scaling_srinivas=True)
        self._ucb = UCB(beta=initial_beta, update_beta=update_beta, beta_scaling_srinivas=True)

    def should_stage_stop(
        self, runhistory: RunHistory, budgets_in_stage: dict[int, list[int]], scenario: Scenario, stage_info: Stage
    ) -> bool:
        """
        Check if a stage should stop.

        Parameters
        ----------
        runhistory : RunHistory
            Runhistory of the current optimization run
        budgets_in_stage : dict[int, list[int]]
            Budgets in each stage
        scenario : Scenario
            Scenario object
        stage_info : Stage
            Information about the current stage
        """
        # To skip a stage, get the incumbent(s) statistical error on that stage and compare it to the regret of the
        # model on that stage. If the regret is smaller than the statistical error, skip the stage.
        # Get the best configs statistical error
        configs = stage_info.configs
        best_configs = []
        stats = []
        for config in configs:
            trial_keys = runhistory.get_trials(config)

            trial_keys = [
                trial for trial in trial_keys if trial.budget == budgets_in_stage[stage_info.bracket][stage_info.stage]
            ]

            if len(trial_keys) == 0:
                continue

            best_configs.append(config)
            trial_values = [
                runhistory[
                    TrialKey(
                        config_id=runhistory.get_config_id(config),
                        instance=trial.instance,
                        seed=trial.seed,
                        budget=trial.budget,
                    )
                ]
                for trial in trial_keys
            ]

            performance = np.mean([trial_value.cost for trial_value in trial_values])
            if self._statistical_error_field_name is not None:
                error = np.mean(
                    [trial_value.additional_info[self._statistical_error_field_name] for trial_value in trial_values]
                )
            else:
                assert self._statistical_error_threshold is not None
                error = self._statistical_error_threshold

            stats.append((performance, error, config))
        # Sort the configs by their performance
        stats.sort(key=lambda trial: trial[0])

        if self._statistical_error_estimation_only_incumbent:
            selected_amount = 1
        else:
            assert self._statistical_error_config_estimation_percentage is not None
            selected_amount = int(len(stats) * self._statistical_error_config_estimation_percentage)

        statistical_error = float(np.mean([stat[1] for stat in stats[:selected_amount]]))

        # Get the regret of the model on that budget
        # Get data
        encoder = RunHistoryEncoder(scenario)
        encoder.runhistory = runhistory
        x, y = encoder.transform(budget_subset=[budgets_in_stage[stage_info.bracket][stage_info.stage]])

        # Train model
        from smac.facade.blackbox_facade import BlackBoxFacade

        model = BlackBoxFacade.get_model(
            scenario=scenario,
        )
        model.train(x, y)

        # Get lcb and ucb
        self._lcb.update(model=model, num_data=len(x))
        self._ucb.update(model=model, num_data=len(x))
        from smac.callback import StoppingCallback

        min_lcb, min_ucb = StoppingCallback.compute_min_lcb_ucb(
            ucb=self._ucb,
            lcb=self._lcb,
            n_points_lcb=1000,
            configs=best_configs,
            configspace=scenario.configspace,
        )
        regret = min_ucb - min_lcb
        stop = statistical_error > regret

        for callback in self._callbacks:
            callback.log(
                min_ucb=min_ucb,
                min_lcb=min_lcb,
                statistical_error=statistical_error,
                regret=regret,
                triggered=stop,
                stage_info=stage_info,
            )

        return stop
