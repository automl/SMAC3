import time
from multiprocessing.sharedctypes import Value
from unittest import mock

import numpy as np
import pytest
from ConfigSpace import Configuration

from smac.intensifier.successive_halving import SuccessiveHalving
from smac.intensifier.successive_halving_worker import SuccessiveHalvingWorker
from smac.runhistory import RunHistory, TrialInfo, TrialInfoIntent, TrialValue
from smac.runner.abstract_runner import StatusType
from smac.runner.target_function_runner import TargetFunctionRunner
from smac.stats import Stats

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def evaluate_challenger(
    trial_info: TrialInfo,
    target_function: TargetFunctionRunner,
    stats: Stats,
    runhistory: RunHistory,
    force_update=False,
):
    """
    Wrapper over challenger evaluation

    SMBO objects handles run history now, but to keep
    same testing functionality this function is a small
    wrapper to launch the taf and add it to the history
    """
    # evaluating configuration
    trial_info, result = target_function.run_wrapper(
        trial_info=trial_info,
    )

    stats._target_function_walltime_used += float(result.time)
    stats._finished += 1

    runhistory.add(
        config=trial_info.config,
        cost=result.cost,
        time=result.time,
        status=result.status,
        instance=trial_info.instance,
        seed=trial_info.seed,
        budget=trial_info.budget,
        force_update=force_update,
    )
    stats._n_configs = len(runhistory.config_ids)

    return result


def target_from_trial_info(trial_info: TrialInfo):
    value_from_config = sum([a for a in trial_info.config.get_dictionary().values() if not isinstance(a, str)])
    return TrialValue(
        cost=value_from_config,
        time=0.5,
        status=StatusType.SUCCESS,
        starttime=time.time(),
        endtime=time.time() + 1,
        additional_info={},
    )


@pytest.fixture
def SH(make_scenario, make_stats, configspace_small):
    scenario = make_scenario(
        configspace_small,
        use_instances=True,
        n_instances=3,
        deterministic=False,
        min_budget=2,
        max_budget=5,
    )
    stats = make_stats(scenario)
    intensifier = SuccessiveHalving(scenario=scenario, eta=2, n_seeds=2)
    intensifier._stats = stats

    return intensifier


@pytest.fixture
def _SH(SH):
    return SuccessiveHalvingWorker(SH)


@pytest.fixture
def make_sh_worker(make_scenario, make_stats, configspace_small):
    def _make(
        deterministic=False,
        min_budget=2,
        max_budget=5,
        eta=2,
        n_instances=3,
        n_seeds=1,
        min_challenger=1,
        instance_order="shuffle_once",
        incumbent_selection="highest_executed_budget",
        _all_budgets=None,
        _n_configs_in_stage=None,
    ):
        scenario = make_scenario(
            configspace_small,
            use_instances=True,
            n_instances=n_instances,
            deterministic=deterministic,
            min_budget=min_budget,
            max_budget=max_budget,
        )
        stats = make_stats(scenario)
        sh = SuccessiveHalving(
            scenario=scenario,
            instance_order=instance_order,
            incumbent_selection=incumbent_selection,
            min_challenger=min_challenger,
            eta=eta,
            n_seeds=n_seeds,
        )
        sh._stats = stats

        return SuccessiveHalvingWorker(
            sh,
            _all_budgets=_all_budgets,
            _n_configs_in_stage=_n_configs_in_stage,
        )

    return _make


@pytest.fixture
def make_target_function():
    def _make(scenario, func, required_arguments=[]):
        return TargetFunctionRunner(
            target_function=func,
            scenario=scenario,
            required_arguments=required_arguments,
        )

    return _make


@pytest.fixture
def configs(configspace_small):
    configs = configspace_small.sample_configuration(20)
    return (configs[16], configs[15], configs[2], configs[3])


def test_init(SH):
    """Makes sure that a proper _SH is created"""

    # We initialize the SH with zero intensifier_instances
    assert len(SH._intensifier_instances) == 0

    # Add an instance to check the _SH initialization
    assert SH._add_new_instance(n_workers=1)

    # Parameters properly passed to _SH
    assert len(SH._instance_seed_pairs) == 6
    assert SH._min_budget == 2
    assert SH._max_budget == 5


def test_process_results_via_sourceid(SH, runhistory, configs):
    """Makes sure source id is honored when deciding
    which _SH will consume the result/trial_info"""
    # Mock the _SH so we can make sure the correct item is passed
    for i in range(10):
        SH._intensifier_instances[i] = mock.Mock()

    # randomly create trial_infos and push into SH. Then we will make
    # sure they got properly allocated
    for i in np.random.choice(list(range(10)), 30):
        trial_info = TrialInfo(
            config=configs[0],
            instance="i1",
            seed=0,
            budget=None,
            source=i,
        )

        # make sure results aren't messed up via magic variable
        # That is we check only the proper _SH has this
        magic = time.time()

        trial_value = TrialValue(
            cost=1,
            time=0.5,
            status=StatusType.SUCCESS,
            starttime=1,
            endtime=2,
            additional_info=magic,
        )
        SH.process_results(
            trial_info=trial_info,
            incumbent=None,
            runhistory=runhistory,
            time_bound=None,
            trial_value=trial_value,
            log_trajectory=False,
        )

        # Check the call arguments of each sh instance and make sure
        # it is the correct one

        # First the expected one
        assert SH._intensifier_instances[i].process_results.call_args[1]["trial_info"] == trial_info
        assert SH._intensifier_instances[i].process_results.call_args[1]["trial_value"] == trial_value

        all_other_trial_infos, all_other_results = [], []
        for j, item in enumerate(SH._intensifier_instances):
            # Skip the expected _SH
            if i == j:
                continue
            if SH._intensifier_instances[j].process_results.call_args is None:
                all_other_trial_infos.append(None)
            else:
                all_other_trial_infos.append(SH._intensifier_instances[j].process_results.call_args[1]["trial_info"])
                all_other_results.append(SH._intensifier_instances[j].process_results.call_args[1]["trial_value"])

        assert trial_info not in all_other_trial_infos
        assert trial_value not in all_other_results


def test_get_next_trial_single_SH(SH, runhistory, configs):
    """Makes sure that a single _SH returns a valid config"""

    challengers = configs[:4]
    for i in range(30):
        intent, trial_info = SH.get_next_trial(
            challengers=challengers,
            incumbent=None,
            get_next_configurations=None,
            runhistory=runhistory,
            n_workers=1,
        )

        # Regenerate challenger list
        challengers = [c for c in challengers if c != trial_info.config]

        if intent == TrialInfoIntent.WAIT:
            break

        # Add the config to rh in order to make SH aware that this
        # config/instance was launched
        runhistory.add(
            config=trial_info.config,
            cost=10,
            time=0.0,
            status=StatusType.RUNNING,
            instance=trial_info.instance,
            seed=trial_info.seed,
            budget=trial_info.budget,
        )

    # We should not create more _SH intensifier_instances
    assert len(SH._intensifier_instances) == 1

    # We are running with:
    # 'all_budgets': array([2.5, 5. ]) -> 2 intensifier_instances per config top
    # 'n_configs_in_stage': [2.0, 1.0],
    # This means we run int(2.5) + 2.0 = 4 runs before waiting
    assert i == 4


def test_get_next_trial_dual_SH(SH, runhistory, configs):
    """Makes sure that two  _SH can properly coexist and tag
    trial_info properly"""

    # Everything here will be tested with a single _SH
    challengers = configs[:4]
    for i in range(30):
        intent, trial_info = SH.get_next_trial(
            challengers=challengers,
            incumbent=None,
            get_next_configurations=None,
            runhistory=runhistory,
            n_workers=2,
        )

        # Regenerate challenger list
        challengers = [c for c in challengers if c != trial_info.config]

        # Add the config to rh in order to make SH aware that this
        # config/instance was launched
        if intent == TrialInfoIntent.WAIT:
            break
        runhistory.add(
            config=trial_info.config,
            cost=10,
            time=0.0,
            status=StatusType.RUNNING,
            instance=trial_info.instance,
            seed=trial_info.seed,
            budget=trial_info.budget,
        )

    # We create a second sh intensifier_instances as after 4 runs, the _SH
    # number zero needs to wait
    assert len(SH._intensifier_instances) == 2

    # We are running with:
    # 'all_budgets': array([2.5, 5. ]) -> 2 intensifier_instances per config top
    # 'n_configs_in_stage': [2.0, 1.0],
    # This means we run int(2.5) + 2.0 = 4 runs before waiting
    # But we have 2 successive halvers now!
    assert i == 8


def test_add_new_instance(SH):
    """Test whether we can add a _SH and when we should not"""

    # By default we do not create a _SH
    # test adding the first instance!
    assert len(SH._intensifier_instances) == 0
    assert SH._add_new_instance(n_workers=1)
    assert len(SH._intensifier_instances) == 1
    assert isinstance(SH._intensifier_instances[0], SuccessiveHalvingWorker)
    # A second call should not add a new _SH
    assert not SH._add_new_instance(n_workers=1)

    # We try with 2 _SH active

    # We effectively return true because we added a new _SH
    assert SH._add_new_instance(n_workers=2)

    assert len(SH._intensifier_instances) == 2
    assert isinstance(SH._intensifier_instances[1], SuccessiveHalvingWorker)

    # Trying to add a third one should return false
    assert not SH._add_new_instance(n_workers=2)
    assert len(SH._intensifier_instances) == 2


def _exhaust_run_and_get_incumbent(SH, runhistory, configs, n_workers=2):
    """
    Runs all provided configs on all intensifier_instances and return the incumbent
    as a nice side effect runhistory/stats are properly filled
    """
    challengers = configs[:4]
    incumbent = None
    inc_perf = None
    for i in range(100):
        try:
            intent, trial_info = SH.get_next_trial(
                challengers=challengers,
                incumbent=None,
                get_next_configurations=None,
                runhistory=runhistory,
                n_workers=n_workers,
            )
        except ValueError as e:
            # Get configurations until you run out of them
            print(e)
            break

        # Regenerate challenger list
        challengers = [c for c in challengers if c != trial_info.config]

        if intent == TrialInfoIntent.WAIT:
            break

        trial_value = target_from_trial_info(trial_info)
        runhistory.add(
            config=trial_info.config,
            cost=trial_value.cost,
            time=trial_value.time,
            status=trial_value.status,
            instance=trial_info.instance,
            seed=trial_info.seed,
            budget=trial_info.budget,
        )
        incumbent, inc_perf = SH.process_results(
            trial_info=trial_info,
            incumbent=incumbent,
            runhistory=runhistory,
            time_bound=100.0,
            trial_value=trial_value,
            log_trajectory=False,
        )

    return incumbent, inc_perf


def test_parallel_same_as_serial_SH(SH, _SH, configs):
    """Makes sure we behave the same as a serial run at the end"""
    runhistory1 = RunHistory()
    runhistory2 = RunHistory()

    incumbent, inc_perf = _exhaust_run_and_get_incumbent(_SH, runhistory1, configs)

    # Just to make sure nothing has changed from the _SH side to make
    # this check invalid:
    # We add config values, so config x with y and z should be the lesser cost
    assert incumbent == configs[0]
    # assert inc_perf == 7.0

    # Do the same for SH, but have multiple _SH in there
    # _add_new_instance returns true if it was able to add a new _SH
    # We call this method twice because we want 2 workers
    assert SH._add_new_instance(n_workers=2)
    assert SH._add_new_instance(n_workers=2)
    incumbent_psh, inc_perf_psh = _exhaust_run_and_get_incumbent(SH, runhistory2, configs)
    assert incumbent == incumbent_psh

    # This makes sure there is a single incumbent in SH
    assert inc_perf == inc_perf_psh

    # We don't want to loose any configuration, and particularly
    # we want to make sure the values of _SH to SH match
    assert len(runhistory1._data) == len(runhistory2._data)

    # We are comparing exhausted single vs parallel successive
    # halving runs. The number and type of configs should be the same
    # and is enforced as a dictionary key argument check. The number
    # of runs will be different ParallelSuccesiveHalving has 2 _SH intensifier_instances
    # yet we make sure that after exhaustion, the budgets a config was run
    # should match
    configs_sh_rh = {}
    for k, v in runhistory1._data.items():
        config_sh = runhistory1.ids_config[k.config_id]
        if config_sh not in configs_sh_rh:
            configs_sh_rh[config_sh] = []
        if v.cost not in configs_sh_rh[config_sh]:
            configs_sh_rh[config_sh].append(v.cost)

    configs_psh_rh = {}
    for k, v in runhistory2._data.items():
        config_psh = runhistory2.ids_config[k.config_id]
        if config_psh not in configs_psh_rh:
            configs_psh_rh[config_psh] = []
        if v.cost not in configs_psh_rh[config_psh]:
            configs_psh_rh[config_psh].append(v.cost)

    # If this dictionaries are equal it means we have all configs
    # and the values track the numbers and actual cost!
    assert configs_sh_rh == configs_psh_rh


def test_init_1(make_sh_worker):
    """Test parameter initializations for successive halving - instance as budget."""
    _SH = make_sh_worker(deterministic=False, min_budget=None, max_budget=None, n_seeds=2)

    assert len(_SH._successive_halving._instance_seed_pairs) == 6  # since instance-seed pairs
    assert len(_SH._successive_halving._instances) == 3
    assert _SH._successive_halving._min_budget == 1
    assert _SH._successive_halving._max_budget == 6
    assert _SH._n_configs_in_stage == [4.0, 2.0, 1.0]
    assert _SH._successive_halving._instance_as_budget
    assert _SH._successive_halving._repeat_configs


def test_init_2(make_sh_worker):
    """Test parameter initialiations for successive halving - real-valued budget."""
    _SH = make_sh_worker(deterministic=False, min_budget=1, max_budget=10, n_instances=1, n_seeds=10)

    assert len(_SH._successive_halving._instance_seed_pairs) == 10  # since instance-seed pairs
    assert _SH._successive_halving._min_budget == 1
    assert _SH._successive_halving._max_budget == 10
    assert _SH._n_configs_in_stage == [8.0, 4.0, 2.0, 1.0]
    assert list(_SH._all_budgets) == [1.25, 2.5, 5.0, 10.0]
    assert _SH._successive_halving._instance_as_budget
    assert _SH._successive_halving._repeat_configs

    # Max budget > instance seed pairs
    with pytest.raises(ValueError):
        _SH = make_sh_worker(deterministic=False, min_budget=1, max_budget=10, n_instances=1, n_seeds=1)


def test_init_3(make_sh_worker):
    """Test parameter initialiations for successive halving - real-valued budget, high initial budget."""
    # In case of deterministic, only one seed is used
    with pytest.raises(ValueError):
        _SH = make_sh_worker(deterministic=True, min_budget=9, max_budget=10, n_instances=1, n_seeds=10)

    _SH = make_sh_worker(deterministic=True, min_budget=9, max_budget=10, n_instances=10, n_seeds=1)
    assert len(_SH._successive_halving._instance_seed_pairs) == 10  # since instance-seed pairs
    assert _SH._successive_halving._min_budget == 9
    assert _SH._successive_halving._max_budget == 10
    assert _SH._n_configs_in_stage == [1.0]
    assert list(_SH._all_budgets) == [10.0]
    assert _SH._successive_halving._instance_as_budget
    assert not _SH._successive_halving._repeat_configs


def test_init_4(make_sh_worker):
    """
    Test wrong parameter initializations for successive halving
    """
    # eta < 1
    with pytest.raises(ValueError, match="The parameter `eta` must be greater than 1."):
        make_sh_worker(deterministic=True, min_budget=None, max_budget=None, n_instances=1, n_seeds=1, eta=0)

    # max budget > instance-seed pairs
    with pytest.raises(ValueError, match="Max budget.*"):
        make_sh_worker(deterministic=False, min_budget=1, max_budget=5, n_instances=3, n_seeds=1)


def test_top_k_1(make_sh_worker, runhistory, configs):
    """
    test _top_k() for configs with same instance-seed-budget keys
    """
    intensifier = make_sh_worker(n_instances=2, min_budget=1, max_budget=4, n_seeds=2)
    config1 = configs[0]
    config2 = configs[1]
    config3 = configs[2]
    config4 = configs[3]

    runhistory.add(
        config=config1,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
    )
    runhistory.add(
        config=config1,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance="i2",
        seed=None,
        additional_info=None,
    )
    runhistory.add(
        config=config2,
        cost=2,
        time=2,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
    )
    runhistory.add(
        config=config2,
        cost=2,
        time=2,
        status=StatusType.SUCCESS,
        instance="i2",
        seed=None,
        additional_info=None,
    )
    runhistory.add(
        config=config3,
        cost=3,
        time=3,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
    )
    runhistory.add(
        config=config3,
        cost=3,
        time=3,
        status=StatusType.SUCCESS,
        instance="i2",
        seed=None,
        additional_info=None,
    )
    runhistory.add(
        config=config4,
        cost=0.5,
        time=0.5,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
    )
    runhistory.add(
        config=config4,
        cost=0.5,
        time=0.5,
        status=StatusType.SUCCESS,
        instance="i2",
        seed=None,
        additional_info=None,
    )
    conf = intensifier._top_k(
        configs=configs[:4],
        k=2,
        runhistory=runhistory,
    )

    # Check that config4 is also before config1 (as it has the lower cost)
    assert conf == [config4, config1]


def test_top_k_2(make_sh_worker, runhistory, configs):
    """Test _top_k() for configs with different instance-seed-budget keys"""
    intensifier = make_sh_worker(n_instances=2, min_budget=1, max_budget=4, n_seeds=2)
    config1 = configs[0]
    config2 = configs[1]
    config3 = configs[2]

    runhistory.add(
        config=config1,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
    )
    runhistory.add(
        config=config2,
        cost=10,
        time=10,
        status=StatusType.SUCCESS,
        instance="i2",
        seed=None,
        additional_info=None,
    )

    with pytest.raises(ValueError, match="Can not compare configs"):
        intensifier._top_k(
            configs=[config2, config1, config3],
            k=1,
            runhistory=runhistory,
        )


def test_top_k_3(make_sh_worker, runhistory, configs):
    """Test _top_k() for not enough configs to generate for the next budget"""
    intensifier = make_sh_worker(n_instances=1, min_budget=1, max_budget=4, n_seeds=4)
    config1 = configs[0]
    config2 = configs[1]

    runhistory.add(
        config=config1,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
    )
    runhistory.add(
        config=config2,
        cost=1,
        time=1,
        status=StatusType.CRASHED,
        instance="i1",
        seed=None,
        additional_info=None,
    )
    configs = intensifier._top_k(configs=[config1], k=2, runhistory=runhistory)

    # top_k should return whatever configuration is possible
    assert configs == [config1]


def test_top_k_4(make_sh_worker, runhistory, configs):
    """Test _top_k() for not enough configs to generate for the next budget"""
    intensifier = make_sh_worker(n_instances=1, min_budget=1, max_budget=10, n_seeds=10, eta=2, min_challenger=1)
    config1 = configs[0]
    config2 = configs[1]
    config3 = configs[2]
    config4 = configs[3]

    intensifier._update_stage(runhistory)
    runhistory.add(
        config=config1,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        budget=1,
        additional_info=None,
    )
    runhistory.add(
        config=config2,
        cost=1,
        time=1,
        status=StatusType.DONOTADVANCE,
        instance="i1",
        seed=None,
        budget=1,
        additional_info=None,
    )
    runhistory.add(
        config=config3,
        cost=1,
        time=1,
        status=StatusType.DONOTADVANCE,
        instance="i1",
        seed=None,
        budget=1,
        additional_info=None,
    )
    runhistory.add(
        config=config4,
        cost=1,
        time=1,
        status=StatusType.DONOTADVANCE,
        instance="i1",
        seed=None,
        budget=1,
        additional_info=None,
    )
    intensifier._success_challengers.add(config1)
    intensifier._failed_challengers.add(config2)
    intensifier._failed_challengers.add(config3)
    intensifier._failed_challengers.add(config4)
    intensifier._update_stage(runhistory)
    assert intensifier._failed_challenger_offset == 3  # We miss three challenger for this round

    configs = intensifier._top_k(configs=[config1], k=2, runhistory=runhistory)
    assert configs == [config1]

    runhistory.add(
        config=config1,
        cost=1,
        time=1,
        status=StatusType.DONOTADVANCE,
        instance="i1",
        seed=None,
        budget=intensifier._all_budgets[1],
        additional_info=None,
    )
    intensifier._failed_challengers.add(config2)
    intensifier._update_stage(runhistory)
    assert intensifier._stage == 0  # Going back, since there are not enough to advance


def test_get_next_trial_1(make_sh_worker, runhistory, configs):
    """
    test get_next_trial for a presently running configuration
    """

    def target(x):
        return 1

    intensifier = make_sh_worker(deterministic=False, n_instances=2, min_budget=1, max_budget=2, n_seeds=1, eta=2)
    target_function = TargetFunctionRunner(intensifier._scenario, target)
    config1 = configs[0]
    config2 = configs[1]

    # next challenger from a list
    intent, trial_info = intensifier.get_next_trial(
        challengers=[config1],
        get_next_configurations=None,
        runhistory=runhistory,
        incumbent=None,
    )
    runhistory.add(
        config=trial_info.config,
        instance=trial_info.instance,
        seed=trial_info.seed,
        budget=trial_info.budget,
        cost=10,
        time=1,
        status=StatusType.RUNNING,
        additional_info=None,
    )
    assert trial_info.config == config1
    assert intensifier._new_challenger

    # In the parallel scenario, we cannot wait for a configuration
    # to be evaluated before moving to the next configuration in the same
    # stage. That is, for this example we will have n_configs_in_stage=[2, 1]
    # with all_budgets=[1. 2.]. In other words, in this stage we
    # will have 2 configs each with 1 instance.
    intent, trial_info_new = intensifier.get_next_trial(
        challengers=[config2],
        get_next_configurations=None,
        runhistory=runhistory,
        incumbent=None,
    )
    runhistory.add(
        config=trial_info_new.config,
        instance=trial_info_new.instance,
        seed=trial_info_new.seed,
        budget=trial_info_new.budget,
        cost=10,
        time=1,
        status=StatusType.RUNNING,
        additional_info=None,
    )
    assert trial_info_new.config == config2
    assert intensifier._running_challenger == trial_info_new.config
    assert intensifier._new_challenger

    # evaluating configuration
    assert trial_info.config is not None
    trial_value = evaluate_challenger(trial_info, target_function, intensifier._stats, runhistory)
    inc, inc_value = intensifier.process_results(
        trial_info=trial_info,
        incumbent=None,
        runhistory=runhistory,
        time_bound=np.inf,
        trial_value=trial_value,
        log_trajectory=False,
    )

    # We already launched trial_info_new. We expect 2 configs each with 1 seed/instance
    # 1 has finished and already processed. We have not even run trial_info_new
    # So we cannot advance to a new stage
    intent, trial_info = intensifier.get_next_trial(
        challengers=[config2], get_next_configurations=None, incumbent=inc, runhistory=runhistory
    )
    assert trial_info.config is None
    assert intent == TrialInfoIntent.WAIT
    assert len(intensifier._success_challengers) == 1
    assert intensifier._new_challenger


def test_get_next_trial_2(make_sh_worker, configs, runhistory):
    """
    test get_next_trial for higher stages of SH iteration
    """
    intensifier = make_sh_worker(deterministic=True, n_instances=2, min_budget=1, max_budget=2, n_seeds=1, eta=2)
    config1 = configs[0]

    intensifier._update_stage(runhistory=None)
    intensifier._stage += 1
    intensifier._configs_to_run = [config1]

    # next challenger should come from configs to run
    intent, trial_info = intensifier.get_next_trial(
        challengers=None,
        get_next_configurations=None,
        runhistory=runhistory,
        incumbent=None,
    )
    assert trial_info.config == config1
    assert len(intensifier._configs_to_run) == 0
    assert not intensifier._new_challenger


def test_update_stage(make_sh_worker, runhistory, configs):
    """Test update_stage - initializations for all tracking variables."""
    intensifier = make_sh_worker(deterministic=True, n_instances=2, min_budget=1, max_budget=2, n_seeds=1, eta=2)
    config1 = configs[0]
    config2 = configs[1]

    # First stage update
    # Change in 2.0: Workers are initialized with stage 0 directly
    # intensifier._update_stage(runhistory=None)

    assert intensifier._stage == 0
    assert intensifier._sh_iters == 0
    assert intensifier._running_challenger is None
    assert intensifier._success_challengers == set()

    # higher stages
    runhistory.add(config1, 1, 1, StatusType.SUCCESS)
    runhistory.add(config2, 2, 2, StatusType.SUCCESS)
    intensifier._success_challengers = {config1, config2}
    intensifier._update_stage(runhistory=runhistory)

    assert intensifier._stage == 1
    assert intensifier._sh_iters == 0
    assert intensifier._configs_to_run == [config1]

    # next iteration
    intensifier._success_challengers = {config1}
    intensifier._update_stage(runhistory=runhistory)

    assert intensifier._stage == 0
    assert intensifier._sh_iters == 1
    assert isinstance(intensifier._configs_to_run, list)
    assert len(intensifier._configs_to_run) == 0


'''
# @unittest.mock.patch.object(_SuccessiveHalving, "_top_k")
def test_update_stage_2(make_sh_worker, runhistory, configs):
    """
    test update_stage - everything good is in state do not advance
    """

    intensifier = make_sh_worker(deterministic=True, n_instances=0, min_budget=1, max_budget=4, n_seeds=1, eta=2)
    config1 = configs[0]
    config2 = configs[1]
    config3 = configs[2]
    config4 = configs[3]

    # update variables
    intensifier._update_stage(runhistory=runhistory)

    intensifier._success_challengers.add(config1)
    intensifier._success_challengers.add(config2)
    intensifier.do_not_advance_challengers.add(config3)
    intensifier.do_not_advance_challengers.add(config4)

    # top_k_mock.return_value = [config1, config3]
    intensifier.return_value = [config1, config3]

    # Test that we update the stage as there is one configuration advanced to the next budget
    assert intensifier._stage == 0
    intensifier._update_stage(runhistory=runhistory)
    assert intensifier._stage == 1
    assert intensifier._configs_to_run == [config1]
    assert intensifier.fail_chal_offset == 1
    assert len(intensifier._success_challengers) == 0
    assert len(intensifier.do_not_advance_challengers) == 0

    intensifier = make_sh_worker(deterministic=True, n_instances=0, min_budget=1, max_budget=4, eta=2)

    # update variables
    intensifier._update_stage(runhistory=runhistory)

    intensifier._success_challengers.add(config1)
    intensifier._success_challengers.add(config2)
    intensifier.do_not_advance_challengers.add(config3)
    intensifier.do_not_advance_challengers.add(config4)

    intensifier.return_value = [config3, config4]
    # top_k_mock.return_value = [config3, config4]

    # Test that we update the stage as there is no configuration advanced to the next budget
    assert intensifier._stage == 0
    intensifier._update_stage(runhistory=runhistory)
    assert intensifier._stage == 0
    assert intensifier._configs_to_run == []
    assert intensifier.fail_chal_offset == 0
    assert len(intensifier._success_challengers) == 0
    assert len(intensifier.do_not_advance_challengers) == 0

    # top_k_mock.return_value = []
    intensifier.return_value = []
'''


def test_evaluate_challenger_1(make_sh_worker, make_target_function, runhistory, configs):
    """Test evaluate_challenger with quality objective & real-valued budget."""

    def target(x: Configuration, seed: int, budget: float):
        return 0.1 * budget

    config1 = configs[0]
    config2 = configs[1]
    config3 = configs[2]

    # instances = [None]???
    intensifier = make_sh_worker(deterministic=True, n_instances=0, min_budget=0.25, max_budget=0.5, eta=2)
    intensifier._update_stage(runhistory=None)

    target_function = make_target_function(intensifier._scenario, target, required_arguments=["seed", "budget"])

    runhistory.add(
        config=config1,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        seed=0,
        budget=0.5,
    )
    runhistory.add(
        config=config2,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        seed=0,
        budget=0.25,
    )
    runhistory.add(
        config=config3,
        cost=2,
        time=1,
        status=StatusType.SUCCESS,
        seed=0,
        budget=0.25,
    )

    intensifier._success_challengers = {config2, config3}
    intensifier._update_stage(runhistory=runhistory)

    intent, trial_info = intensifier.get_next_trial(
        challengers=[config1],
        get_next_configurations=None,
        incumbent=config1,
        runhistory=runhistory,
    )
    trial_value = evaluate_challenger(trial_info, target_function, intensifier._stats, runhistory)
    inc, inc_value = intensifier.process_results(
        trial_info=trial_info,
        incumbent=config1,
        runhistory=runhistory,
        time_bound=np.inf,
        trial_value=trial_value,
    )

    assert inc == config2
    assert inc_value == 0.05
    assert list(runhistory._data.keys())[-1].config_id == runhistory.config_ids[config2]
    assert intensifier._stats.incumbent_changed == 1


def test_incumbent_selection_default(make_sh_worker, make_target_function, runhistory, configs):
    """Test _compare_config for default incumbent selection design (highest budget so far)."""

    config1 = configs[0]
    config2 = configs[1]
    config3 = configs[2]
    config4 = configs[3]

    # instances = [None]???
    intensifier = make_sh_worker(deterministic=True, n_instances=2, min_budget=1, max_budget=2, eta=2)
    intensifier._stage = 0
    # intensifier._update_stage(runhistory=None)
    # SH considers challenger as incumbent in first run in evaluate_challenger
    runhistory.add(
        config=config1,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
        budget=1,
    )
    inc = intensifier._compare_configs(
        challenger=config1,
        incumbent=config1,
        runhistory=runhistory,
        log_trajectory=False,
    )
    assert inc == config1
    runhistory.add(
        config=config1,
        cost=1,
        time=1,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
        budget=2,
    )
    inc = intensifier._compare_configs(challenger=config1, incumbent=inc, runhistory=runhistory, log_trajectory=False)
    assert inc == config1

    # Adding a worse configuration
    runhistory.add(
        config=config2,
        cost=2,
        time=2,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
        budget=1,
    )
    inc = intensifier._compare_configs(challenger=config2, incumbent=inc, runhistory=runhistory, log_trajectory=False)
    assert inc == config1
    runhistory.add(
        config=config2,
        cost=2,
        time=2,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
        budget=2,
    )
    inc = intensifier._compare_configs(challenger=config2, incumbent=inc, runhistory=runhistory, log_trajectory=False)
    assert inc == config1

    # Adding a better configuration, but the incumbent will only be changed on budget=2
    runhistory.add(
        config=config3,
        cost=0.5,
        time=3,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
        budget=1,
    )
    inc = intensifier._compare_configs(challenger=config3, incumbent=inc, runhistory=runhistory, log_trajectory=False)
    assert inc == config1
    runhistory.add(
        config=config3,
        cost=0.5,
        time=3,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
        budget=2,
    )
    inc = intensifier._compare_configs(challenger=config3, incumbent=inc, runhistory=runhistory, log_trajectory=False)
    assert inc == config3

    intensifier = make_sh_worker(deterministic=True, n_instances=5, min_budget=1, eta=2)
    intensifier._stage = 0

    # Adding a better configuration, but the incumbent will only be changed on budget=2
    runhistory.add(
        config=config4,
        cost=0.1,
        time=3,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
        budget=1,
    )
    inc = intensifier._compare_configs(challenger=config4, incumbent=inc, runhistory=runhistory, log_trajectory=False)
    assert inc == config3
    runhistory.add(
        config=config4,
        cost=0.1,
        time=3,
        status=StatusType.SUCCESS,
        instance="i1",
        seed=None,
        additional_info=None,
        budget=2,
    )
    inc = intensifier._compare_configs(challenger=config4, incumbent=inc, runhistory=runhistory, log_trajectory=False)
    assert inc == config4


def test_incumbent_selection_designs(make_sh_worker, make_target_function, runhistory, configs):
    """Test _compare_config with different incumbent selection designs."""

    config1 = configs[0]
    config2 = configs[1]
    config3 = configs[2]
    config4 = configs[3]

    # instances = [None]???
    intensifier = make_sh_worker(
        deterministic=True,
        n_instances=2,
        min_budget=1,
        max_budget=2,
        eta=2,
        incumbent_selection="any_budget",
    )
    intensifier._stage = 0

    runhistory.add(
        config=config1,
        instance="i1",
        seed=0,
        budget=None,
        cost=10,
        time=1,
        status=StatusType.SUCCESS,
        additional_info=None,
    )
    runhistory.add(
        config=config1,
        instance="i2",
        seed=0,
        budget=None,
        cost=10,
        time=1,
        status=StatusType.SUCCESS,
        additional_info=None,
    )
    runhistory.add(
        config=config2,
        instance="i1",
        seed=0,
        budget=None,
        cost=5,
        time=1,
        status=StatusType.SUCCESS,
        additional_info=None,
    )

    # incumbent should be config2, since it has the best performance in one of the budgets
    inc = intensifier._compare_configs(
        incumbent=config2,
        challenger=config1,
        runhistory=runhistory,
        log_trajectory=False,
    )
    assert config2 == inc

    # if config1 is incumbent already, it shouldn't change
    inc = intensifier._compare_configs(
        incumbent=config1,
        challenger=config2,
        runhistory=runhistory,
        log_trajectory=False,
    )
    assert config1 == inc

    # select best on highest budget only
    intensifier = make_sh_worker(
        deterministic=True,
        n_instances=4,
        min_budget=1,
        max_budget=4,
        eta=2,
        incumbent_selection="highest_budget",
    )
    intensifier._stage = 0

    # incumbent should not change, since there is no run on the highest budget,
    # though config3 is run on a higher budget
    runhistory.add(
        config=config3,
        instance="i1",
        seed=0,
        budget=None,
        cost=0.5,
        time=1,
        status=StatusType.SUCCESS,
        additional_info=None,
    )
    runhistory.add(
        config=config4,
        instance="i1",
        seed=0,
        budget=None,
        cost=5,
        time=1,
        status=StatusType.SUCCESS,
        additional_info=None,
    )
    inc = intensifier._compare_configs(
        incumbent=config4,
        challenger=config3,
        runhistory=runhistory,
        log_trajectory=False,
    )
    assert config3 == inc
    assert intensifier._stats.incumbent_changed == 0

    # incumbent changes to config3 since that is run on the highest budget
    runhistory.add(
        config=config3,
        instance="i1",
        seed=0,
        budget=None,
        cost=10,
        time=1,
        status=StatusType.SUCCESS,
        additional_info=None,
    )
    inc = intensifier._compare_configs(
        incumbent=config4,
        challenger=config3,
        runhistory=runhistory,
        log_trajectory=False,
    )
    assert config3 == inc


def test_launched_all_configs_for_current_stage(make_sh_worker, make_target_function, runhistory, configs):
    """This check makes sure we can identify when all the current runs (config/instance/seed) pairs for a given stage
    have been launched."""

    config2 = configs[1]
    config3 = configs[2]
    config4 = configs[3]

    # instances = [None]???
    intensifier = make_sh_worker(
        deterministic=True, n_instances=10, min_budget=2, max_budget=10, eta=2, incumbent_selection="any_budget"
    )
    intensifier._stage = 0

    # So there are 2 instances per config.
    # stage=0
    # n_configs_in_stage=[4.0, 2.0, 1.0]
    # all_budgets=[ 2.5  5.  10. ]
    total_configs_in_stage = 4
    instances_per_stage = 2

    # get all configs and add them to the dict
    run_tracker = {}
    challengers = configs[:4]
    for i in range(total_configs_in_stage * instances_per_stage):
        intent, trial_info = intensifier.get_next_trial(
            challengers=challengers,
            get_next_configurations=None,
            runhistory=runhistory,
            incumbent=None,
        )

        # All this runs are valid for this stage
        assert intent == TrialInfoIntent.RUN

        # Remove from the challengers, the launched configs
        challengers = [c for c in challengers if c != trial_info.config]
        run_tracker[(trial_info.config, trial_info.instance, trial_info.seed)] = False
        runhistory.add(
            config=trial_info.config,
            instance=trial_info.instance,
            seed=trial_info.seed,
            budget=trial_info.budget,
            cost=10,
            time=1,
            status=StatusType.RUNNING,
            additional_info=None,
        )

    # This will get us the second instance of config 1
    intent, trial_info = intensifier.get_next_trial(
        challengers=[config2, config3, config4],
        get_next_configurations=None,
        runhistory=runhistory,
        incumbent=None,
    )

    # We have launched all runs, that are expected for this stage
    # not registered any, so for sure we have to wait
    # For all runs to be completed before moving to the next stage
    assert intent == TrialInfoIntent.WAIT


def _exhaust_stage_execution(intensifier, target_function, runhistory, challengers, incumbent):
    """
    Exhaust configuration/instances seed and returns the
    trial_info that were not launched.

    The idea with this procedure is to emulate the fact that some
    configurations will finish while others won't. We need to be
    robust against this scenario
    """
    pending_processing = []
    stage = 0 if not hasattr(intensifier, "stage") else intensifier._stage
    curr_budget = intensifier._all_budgets[stage]
    prev_budget = int(intensifier._all_budgets[stage - 1]) if stage > 0 else 0
    if intensifier._successive_halving._instance_as_budget:
        total_runs = int(curr_budget - prev_budget) * int(intensifier._n_configs_in_stage[stage])
        toggle = np.random.choice([True, False], total_runs).tolist()
        while not np.any(toggle) or not np.any(np.invert(toggle)):
            # make sure we have both true and false!
            toggle = np.random.choice([True, False], total_runs).tolist()
    else:
        # If we directly use the budget, then there are no instances to wait
        # But we still want to mimic pending configurations. That is, we don't
        # advance to the next stage until all configurations are done for a given
        # budget.
        # Here if we do not launch a configuration because toggle was false, is
        # like this configuration never exited as there is only 1 instance in this
        # and if toggle is false, it is never run. So we cannot do a random toggle
        toggle = [False, True, False, True]

    while True:
        intent, trial_info = intensifier.get_next_trial(
            challengers=challengers,
            get_next_configurations=None,
            runhistory=runhistory,
            incumbent=incumbent,
        )

        # Update the challengers
        challengers = [c for c in challengers if c != trial_info.config]

        if intent == TrialInfoIntent.WAIT:
            break

        # Add this configuration as running
        runhistory.add(
            config=trial_info.config,
            instance=trial_info.instance,
            seed=trial_info.seed,
            budget=trial_info.budget,
            cost=1000,
            time=1000,
            status=StatusType.RUNNING,
            additional_info=None,
        )

        if toggle.pop():
            trial_value = evaluate_challenger(
                trial_info, target_function, intensifier._stats, runhistory, force_update=True
            )
            incumbent, inc_value = intensifier.process_results(
                trial_info=trial_info,
                incumbent=incumbent,
                runhistory=runhistory,
                time_bound=np.inf,
                trial_value=trial_value,
                log_trajectory=False,
            )
        else:
            pending_processing.append(trial_info)

        # In case a iteration is done, break
        # This happens if the configs per stage is 1
        if intensifier._iteration_done:
            break

    return pending_processing, incumbent


def test_iteration_done_only_when_all_configs_processed_instance_as_budget(
    make_sh_worker, make_target_function, runhistory, configs
):
    """Makes sure that iteration done for a given stage is asserted ONLY after all
    configurations AND instances are completed, when instance is used as budget."""

    def target(x: Configuration):
        return 1

    config1 = configs[0]

    # instances = [None]???
    intensifier = make_sh_worker(deterministic=True, n_instances=5, min_budget=2, max_budget=5, eta=2)
    intensifier._update_stage(runhistory=None)

    target_function = make_target_function(intensifier._scenario, target)
    target_function.runhistory = runhistory

    # we want to test instance as budget
    assert intensifier._successive_halving._instance_as_budget

    # Run until there are no more configurations to be proposed
    # Skip running some configurations to emulate the fact that runs finish on different time
    # We need this because there was a bug where not all instances had finished, yet
    # the SH instance assumed all configurations finished
    challengers = configs[:4]
    incumbent = None
    pending_processing, incumbent = _exhaust_stage_execution(
        intensifier, target_function, runhistory, challengers, incumbent
    )

    # We have configurations pending, so iteration should NOT be done
    assert not intensifier._iteration_done

    # Make sure we launched all configurations we were meant to:
    # all_budgets=[2.5 5. ] n_configs_in_stage=[2.0, 1.0]
    # We need 2 configurations in the run history
    configurations = set([k.config_id for k, v in runhistory._data.items()])
    assert configurations == {1, 2}
    # We need int(2.5) instances in the run history per config
    config_inst_seed = set([k for k, v in runhistory._data.items()])
    assert len(config_inst_seed) == 4

    # Go to the last stage. Notice that iteration should not be done
    # as we are in stage 1 out of 2
    for trial_info in pending_processing:
        trial_value = evaluate_challenger(
            trial_info, target_function, intensifier._stats, runhistory, force_update=True
        )
        incumbent, inc_value = intensifier.process_results(
            trial_info=trial_info,
            incumbent=config1,
            runhistory=runhistory,
            time_bound=np.inf,
            trial_value=trial_value,
            log_trajectory=False,
        )
    assert not intensifier._iteration_done

    # we transition to stage 1, where the budget is 5
    assert intensifier._stage == 1

    pending_processing, incumbent = _exhaust_stage_execution(
        intensifier,
        target_function,
        runhistory,
        challengers,
        incumbent,
    )

    # Because budget is 5, BUT we previously ran 2 instances in stage 0
    # we expect that the run history will be populated with 3 new instances for 1
    # config more 4 (stage0, 2 config on 2 instances) + 3 (stage1, 1 config 3 instances) = 7
    config_inst_seed = [k for k, v in runhistory._data.items()]
    assert len(config_inst_seed) == 7

    # All new runs should be on the same config
    assert len(set([c.config_id for c in config_inst_seed[4:]])) == 1
    # We need 3 new instance seed pairs
    assert len(set(config_inst_seed[4:])) == 3

    # because there are configurations pending, no iteration should be done
    assert not intensifier._iteration_done

    # Finish the pending runs
    for trial_info in pending_processing:
        trial_value = evaluate_challenger(
            trial_info, target_function, intensifier._stats, runhistory, force_update=True
        )
        incumbent, _ = intensifier.process_results(
            trial_info=trial_info,
            incumbent=incumbent,
            runhistory=runhistory,
            time_bound=np.inf,
            trial_value=trial_value,
            log_trajectory=False,
        )

    # Finally, all stages are done, so iteration should be done!!
    assert intensifier._iteration_done


def test_iteration_done_only_when_all_configs_processed_no_instance_as_budget(
    make_sh_worker, make_target_function, runhistory, configs
):
    """Makes sure that iteration done for a given stage is asserted ONLY after all
    configurations AND instances are completed, when instance is NOT used as budget."""

    def target(x: Configuration, seed, budget):
        return 1

    # instances = [None]???
    intensifier = make_sh_worker(
        deterministic=True,
        n_instances=5,
        min_budget=2,
        max_budget=5,
        eta=2,
    )
    intensifier._update_stage(runhistory=None)

    target_function = make_target_function(
        intensifier._scenario,
        target,
        required_arguments=["seed", "budget"],
    )
    target_function.runhistory = runhistory

    assert intensifier._successive_halving._instance_as_budget

    # Run until there are no more configurations to be proposed
    # Skip running some configurations to emulate the fact that runs finish on different time
    # We need this because there was a bug where not all instances had finished, yet
    # the SH instance assumed all configurations finished
    challengers = configs[:4]
    incumbent = None
    pending_processing, incumbent = _exhaust_stage_execution(
        intensifier, target_function, runhistory, challengers, incumbent
    )

    # We have configurations pending, so iteration should NOT be done
    assert not intensifier._iteration_done

    # Make sure we launched all configurations we were meant to:
    # all_budgets=[2.5 5. ] n_configs_in_stage=[2.0, 1.0]
    # We need 2 configurations in the run history
    configurations = set([k.config_id for k, v in runhistory._data.items()])
    assert configurations == {1, 2}
    # There is only one instance always -- so we only have 2 configs for 1 instances each
    config_inst_seed = set([k for k, v in runhistory._data.items()])
    assert len(config_inst_seed) == 4

    # Go to the last stage. Notice that iteration should not be done
    # as we are in stage 1 out of 2
    for trial_info in pending_processing:
        trial_value = evaluate_challenger(
            trial_info,
            target_function,
            intensifier._stats,
            runhistory,
            force_update=True,
        )
        incumbent, _ = intensifier.process_results(
            trial_info=trial_info,
            incumbent=incumbent,
            runhistory=runhistory,
            time_bound=np.inf,
            trial_value=trial_value,
            log_trajectory=False,
        )
    assert not intensifier._iteration_done

    # we transition to stage 1, where the budget is 5
    assert intensifier._stage == 1

    pending_processing, incumbent = _exhaust_stage_execution(
        intensifier, target_function, runhistory, challengers, incumbent
    )

    # The next configuration per stage is just one (n_configs_in_stage=[2.0, 1.0])
    # We ran previously 2 configs and with this new, we should have 3 total
    config_inst_seed = [k for k, v in runhistory._data.items()]
    assert len(config_inst_seed) == 7

    # Because it is only 1 config, the iteration is completed
    assert not intensifier._iteration_done


def test_budget_initialization(make_sh_worker, make_target_function, runhistory, configs):
    """Check computing budgets (only for non-instance cases)."""
    intensifier = make_sh_worker(deterministic=True, n_instances=0, min_budget=1, max_budget=81, eta=3)

    assert [1, 3, 9, 27, 81] == intensifier._all_budgets
    assert [81, 27, 9, 3, 1] == intensifier._n_configs_in_stage

    to_check = [
        # minb, maxb, eta, n_configs_in_stage, all_budgets
        [1, 81, 3, [81, 27, 9, 3, 1], [1, 3, 9, 27, 81]],
        [
            1,
            600,
            3,
            [243, 81, 27, 9, 3, 1],
            [2.469135, 7.407407, 22.222222, 66.666666, 200, 600],
        ],
        [1, 100, 10, [100, 10, 1], [1, 10, 100]],
        [
            0.001,
            1,
            3,
            [729, 243, 81, 27, 9, 3, 1],
            [0.001371, 0.004115, 0.012345, 0.037037, 0.111111, 0.333333, 1.0],
        ],
        [
            1,
            1000,
            3,
            [729, 243, 81, 27, 9, 3, 1],
            [
                1.371742,
                4.115226,
                12.345679,
                37.037037,
                111.111111,
                333.333333,
                1000.0,
            ],
        ],
        [
            0.001,
            100,
            10,
            [100000, 10000, 1000, 100, 10, 1],
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        ],
    ]

    for minb, maxb, eta, n_configs_in_stage, all_budgets in to_check:

        intensifier = make_sh_worker(
            deterministic=True,
            n_instances=0,
            min_budget=minb,
            max_budget=maxb,
            eta=eta,
            _all_budgets=all_budgets,
            _n_configs_in_stage=n_configs_in_stage,
        )

        comp_budgets = intensifier._all_budgets
        comp_configs = intensifier._n_configs_in_stage

        assert len(all_budgets) == len(comp_budgets)
        assert comp_budgets[-1] == maxb
        np.testing.assert_array_almost_equal(all_budgets, comp_budgets, decimal=5)

        assert comp_configs[-1] == 1
        assert len(n_configs_in_stage) == len(comp_configs)
        np.testing.assert_array_almost_equal(n_configs_in_stage, comp_configs, decimal=5)
