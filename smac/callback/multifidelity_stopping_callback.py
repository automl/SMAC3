import numpy as np

from smac import RunHistory, Scenario
from smac.acquisition.function import LCB, UCB
from smac.intensifier.stage_information import Stage
from smac.runhistory import TrialKey
from smac.runhistory.encoder import RunHistoryEncoder


def should_stage_stop(
    runhistory: RunHistory, _budgets_in_stage: dict[int, list[int]], _scenario: Scenario, stage_info: Stage
) -> bool:
    """
    Check if a stage should stop.

    Parameters
    ----------
    runhistory : RunHistory
        Runhistory of the current optimization run
    _budgets_in_stage : dict[int, list[int]]
        Budgets in each stage
    _scenario : Scenario
        Scenario object
    stage_info : Stage
        Information about the current stage
    """
    # TODO: move this info somewhere else
    stat_error_field_name = "statistical_error"
    best_n = "best_config"
    # To skip a stage, get the incumbent(s) statistical error on that stage and compare it to the regret of the
    # model on that stage. If the regret is smaller than the statistical error, skip the stage.
    # Get the best configs statistical error
    rh = runhistory
    configs = stage_info.configs
    # TODO select best configs somehow
    best_configs = []
    stats = []
    for config in configs:
        trial_keys = rh.get_trials(config)

        trial_keys = [
            trial for trial in trial_keys if trial.budget == _budgets_in_stage[stage_info.bracket][stage_info.stage]
        ]

        if len(trial_keys) == 0:
            continue

        best_configs.append(config)
        trial_values = [
            rh[
                TrialKey(
                    config_id=rh.get_config_id(config), instance=trial.instance, seed=trial.seed, budget=trial.budget
                )
            ]
            for trial in trial_keys
        ]

        performance = np.mean([trial_value.cost for trial_value in trial_values])
        error = np.mean([trial_value.additional_info[stat_error_field_name] for trial_value in trial_values])

        stats.append((performance, error, config))
    # Sort the configs by their performance
    stats.sort(key=lambda trial: trial[0])
    # Get the best config
    if best_n == "best_config":
        # TODO select right amount
        select_amount = 1
        statistical_error = np.mean([stat[1] for stat in stats[:select_amount]])
    else:
        raise NotImplementedError()
    # Get the regret of the model on that budget
    # Build model
    encoder = RunHistoryEncoder(_scenario)
    encoder.runhistory = rh
    x, y = encoder.transform(budget_subset=[_budgets_in_stage[stage_info.bracket][stage_info.stage]])
    from smac.facade.blackbox_facade import BlackBoxFacade

    model = BlackBoxFacade.get_model(
        scenario=_scenario,
    )
    model.train(x, y)
    # get lcb and ucb
    initial_beta = 0.1
    update_beta = True
    lcb = LCB(beta=initial_beta, update_beta=update_beta, beta_scaling_srinivas=True)
    ucb = UCB(beta=initial_beta, update_beta=update_beta, beta_scaling_srinivas=True)
    lcb.update(model=model, num_data=len(x))
    ucb.update(model=model, num_data=len(x))
    from smac.callback import StoppingCallback

    min_lcb, min_ucb = StoppingCallback.compute_min_lcb_ucb(
        ucb=ucb,
        lcb=lcb,
        n_points_lcb=1000,
        configs=best_configs,
        configspace=_scenario.configspace,
    )
    regret = min_ucb - min_lcb
    stop = statistical_error > regret
    return stop
