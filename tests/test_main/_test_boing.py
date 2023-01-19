from unittest.mock import patch

import numpy as np
import pytest
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    UniformFloatHyperparameter,
)

import smac
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade,
)
from smac.facade.old.boing_facade import BOinGFacade
from smac.main.boing import subspace_extraction
from smac.model.random_forest.random_forest import RandomForest
from smac.model.utils import check_subspace_points, get_types
from smac.runhistory.runhistory import RunHistory
from smac.runner.abstract_runner import StatusType
from smac.utils import _test_helpers


def test_init(make_scenario):
    scenario = make_scenario(
        configspace=_test_helpers.get_branin_config_space(),
    )
    tae = lambda x: x
    with pytest.raises(ValueError) as excinfo:
        BOinGFacade(scenario=scenario, target_function=tae, model=BlackBoxFacade.get_model(scenario))
        assert excinfo.values == "BOinG only supports RandomForestWithInstances as its global optimizer"

    with pytest.raises(ValueError) as excinfo:
        BOinGFacade(
            scenario=scenario,
            target_function=tae,
            runhistory_encoder=HyperparameterOptimizationFacade.get_runhistory_encoder(scenario),
        )
        assert excinfo.values == "BOinG only supports RunHistory2EPM4CostWithRaw as its rh transformer"

    facade = BOinGFacade(scenario=scenario, target_function=tae, overwrite=True)
    assert not hasattr(facade._optimizer, "turbo_optimizer")

    # facade.do_switching = True
    # facade._init_optimizer()
    # assert hasattr(facade.optimizer, "turbo_optimizer")


def test_chooser_next(make_scenario):
    configspace = _test_helpers.get_branin_config_space()
    scenario = make_scenario(
        configspace=configspace,
    )
    config = scenario.configspace.sample_configuration()
    rh = RunHistory()
    rh.add(config, 10, 10, StatusType.SUCCESS)
    tae = lambda x: x
    facade = BOinGFacade(scenario=scenario, runhistory=rh, target_function=tae, do_switching=False, overwrite=True)
    optimizer = facade.optimizer

    x = next(optimizer.ask())
    # when number of points is not large enough for building a subspace, GP works locally
    assert x.origin == "Local Search"
    for i in range(15):
        config = scenario.configspace.sample_configuration()
        rh.add(config, 10, 10, StatusType.SUCCESS)

    x = next(optimizer.ask())
    # when number of points is already large enough for building a subspace, BOinG takes over
    assert x.origin == "BOinG"

    facade.do_switching = True
    facade._init_optimizer()
    optimizer = facade.optimizer
    optimizer.run_TuRBO = True
    x = next(optimizer.ask())
    assert x.origin == "TuRBO"


def test_do_switching(make_scenario):
    seed = 42
    configspace = _test_helpers.get_branin_config_space()
    scenario = make_scenario(
        configspace=configspace,
    )
    config = scenario.configspace.sample_configuration()
    rh = RunHistory()
    rh.add(config, 10, 10, StatusType.SUCCESS)
    tae = lambda x: x
    turbo_kwargs = {"failure_tol_min": 1, "length_min": 0.6}

    facade = BOinGFacade(
        scenario=scenario,
        runhistory=rh,
        target_function=tae,
        do_switching=True,
        turbo_kwargs=turbo_kwargs,
        overwrite=True,
    )
    optimizer = facade.optimizer

    for i in range(15):
        config = scenario.configspace.sample_configuration()
        rh.add(config, 10, 10, StatusType.SUCCESS)
    config = scenario.configspace.sample_configuration()
    # ensure config is the incumbent
    rh.add(config, 9.99, 10, StatusType.SUCCESS)
    next(optimizer.ask())

    # init an optimal config
    np.testing.assert_allclose(config.get_array(), optimizer.optimal_config)
    assert optimizer.optimal_value == 9.99
    assert 0 == optimizer.failcount_BOinG

    optimizer.failcount_BOinG = 19
    # in this case, prob_to_TurBO becomes 1
    with patch("smac.main.boing.BOinGSMBO." "restart_TuRBOinG") as mk:
        next(optimizer.ask())
        assert optimizer.run_TuRBO is True
        assert mk.called is True

    # switch to TuRBO
    from smac.utils.subspaces.turbo_subspace import TuRBOSubSpace

    for i in range(1000):
        with patch.object(smac.utils.subspaces.turbo_subspace.TuRBOSubSpace, "generate_challengers", return_value=None):
            optimizer.ask()
            while len(optimizer.turbo_optimizer.init_configs) > 0:
                optimizer.turbo_optimizer.init_configs.pop()
        if not optimizer.run_TuRBO:
            break
    # TuRBO will be replaced with BOinG if it cannot find a better value continuously
    assert i < 999

    optimizer.failcount_BOinG = 19
    next(optimizer.ask())

    config = scenario.configspace.sample_configuration()
    rh.add(config, 9.5, 10, StatusType.SUCCESS)
    optimizer.turbo_optimizer.init_configs = []
    for i in range(10):
        next(optimizer.ask())
        if not optimizer.run_TuRBO:
            break
    # one time success and two times failure totally, 3 times evaluations and in this case we have i==2
    assert i == 2


def test_subspace_extraction():
    cs = ConfigurationSpace(0)
    cs.add_hyperparameter(UniformFloatHyperparameter("x0", 0.0, 1.0))
    cs.add_hyperparameter(CategoricalHyperparameter("x1", [0, 1, 2, 3, 4, 5]))

    rf = RandomForest(
        cs,
        num_trees=10,
        ratio_features=1.0,
        min_samples_split=2,
        min_samples_leaf=1,
        seed=0,
    )

    X = np.array([[0.0, 0], [0.2, 1], [0.3, 2], [0.7, 5], [0.6, 3]])

    Y = np.array([0.1, 0.2, 0.7, 0.6, 0.5])

    X_inc = np.array([0.4, 3])
    rf.train(X, Y)
    _, bounds = get_types(cs)

    ss_extraction_kwargs = dict(X=X, challenger=X_inc, model=rf, bounds=bounds, cat_dims=[1], cont_dims=[0])

    num_min = 2
    num_max = 5

    ss_bounds_cont, ss_bounds_cat, ss_indices = subspace_extraction(
        num_min=num_min, num_max=np.inf, **ss_extraction_kwargs
    )
    assert num_min <= sum(ss_indices)
    x_in_ss = check_subspace_points(X_inc, [0], [1], ss_bounds_cont, ss_bounds_cat)
    assert x_in_ss[0]
    ss_indices_re_exam = check_subspace_points(X, [0], [1], ss_bounds_cont, ss_bounds_cat)
    assert sum(ss_indices) == sum(ss_indices_re_exam)

    ss_bounds_cont, ss_bounds_cat, ss_indices = subspace_extraction(
        num_min=num_min, num_max=num_max, **ss_extraction_kwargs
    )
    assert num_min <= sum(ss_indices) <= num_max
    x_in_ss = check_subspace_points(X_inc, [0], [1], ss_bounds_cont, ss_bounds_cat)
    assert x_in_ss[0]
    ss_indices_re_exam = check_subspace_points(X, [0], [1], ss_bounds_cont, ss_bounds_cat)
    assert sum(ss_indices) == sum(ss_indices_re_exam)

    num_max = 3
    ss_bounds_cont, ss_bounds_cat, ss_indices = subspace_extraction(
        num_min=num_min, num_max=num_max, **ss_extraction_kwargs
    )
    assert num_min <= sum(ss_indices) <= num_max
    assert x_in_ss[0]
    ss_indices_re_exam = check_subspace_points(X, [0], [1], ss_bounds_cont, ss_bounds_cat)
    assert sum(ss_indices) == sum(ss_indices_re_exam)
