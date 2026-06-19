from __future__ import annotations
import pytest

from ConfigSpace import ConfigurationSpace, Configuration, Float
import numpy as np

from smac.runhistory.dataclasses import TrialValue
from smac.acquisition.function.confidence_bound import LCB
from smac.initial_design.random_design import RandomInitialDesign
from smac import BlackBoxFacade, HyperparameterOptimizationFacade, Scenario
from smac.main.config_selector import ConfigSelector
from smac.main import config_selector


def test_estimated_config_values_are_trained_by_models(rosenbrock):
    scenario = Scenario(rosenbrock.configspace, n_trials=100, n_workers=2, deterministic=True)
    smac = BlackBoxFacade(
        scenario,
        rosenbrock.train,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        config_selector=ConfigSelector(
            scenario=scenario,
            retrain_after=1,
            batch_sampling_estimation_strategy='no_estimate'
        ),
        initial_design=BlackBoxFacade.get_initial_design(scenario=scenario, n_configs=5),
        acquisition_function=LCB()  # this ensures that we can record the number of data in the acquisition function
    )
    # we first initialize multiple configurations as the starting points

    n_data_in_acq_func = 5
    for _ in range(n_data_in_acq_func):
        info = smac.ask()  # we need the seed from the configuration

        cost = rosenbrock.train(info.config, seed=info.seed, budget=info.budget, instance=info.instance)
        value = TrialValue(cost=cost, time=0.5)

        smac.tell(info, value)

    # for naive approach, no point configuration values is hallucinate
    all_asked_infos = []
    for i in range(3):
        all_asked_infos.append(smac.ask())
        assert smac._acquisition_function._num_data == n_data_in_acq_func

    # each time when we provide a new running configuration, we can estimate the configuration values for new
    # suggestions and use this information to retrain our model. Hence, each time a new point is asked, we should
    # have _num_data +1 for LCB model

    n_data_in_acq_func += 3
    for estimate_strategy in ['CL_max', 'CL_min', 'CL_mean', 'kriging_believer', 'sample']:
        smac._config_selector._batch_sampling_estimation_strategy = estimate_strategy
        for i in range(3):
            all_asked_infos.append(smac.ask())
            assert smac._acquisition_function._num_data == n_data_in_acq_func
            n_data_in_acq_func += 1

    for info in all_asked_infos:
        value = TrialValue(cost=rosenbrock.train(info.config, instance=info.instance, seed=info.seed), )
        smac.tell(info=info, value=value)

    # now we recover to the vanilla approach, in this case, all the evaluations are exact evaluations, the number of
    # data in the runhistory should not increase
    _ = smac.ask()
    assert smac._acquisition_function._num_data == n_data_in_acq_func


@pytest.mark.parametrize("estimation_strategy", ['CL_max', 'CL_min', 'CL_mean', 'kriging_believer', 'sample'])
def test_batch_estimation_methods(rosenbrock, estimation_strategy):
    config_space = rosenbrock.configspace
    scenario = Scenario(config_space, n_trials=100, n_workers=2, deterministic=True)
    config_selector = ConfigSelector(
        scenario=scenario,
        retrain_after=1,
        batch_sampling_estimation_strategy=estimation_strategy
    )
    model = BlackBoxFacade.get_model(scenario=scenario)
    X_evaluated = config_space.sample_configuration(5)
    y_train = np.asarray([rosenbrock.train(x) for x in X_evaluated])
    x_train = np.asarray([x.get_array() for x in X_evaluated])

    model.train(x_train, y_train)

    X_running = np.asarray([x.get_array() for x in config_space.sample_configuration(3)])
    config_selector._model = model

    estimations = config_selector.estimate_running_config_costs(
        X_running, y_train, estimation_strategy=estimation_strategy,
    )
    if estimation_strategy == 'CL_max':
        assert (estimations == y_train.max()).all()
    elif estimation_strategy == 'CL_min':
        assert (estimations == y_train.min()).all()
    elif estimation_strategy == 'CL_mean':
        assert (estimations == y_train.mean()).all()
    else:
        if estimation_strategy == 'kriging_believer':
            assert np.allclose(model.predict_marginalized(X_running)[0], estimations)
        else:
            # for sampling strategy, we simply check if the shape of the two results are the same
            assert np.equal(estimations.shape, (3, 1)).all()
