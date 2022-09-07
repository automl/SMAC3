import time
import pytest
from unittest import mock

import numpy as np

from smac.intensification.hyperband import Hyperband
from smac.intensification.hyperband_worker import HyperbandWorker
from smac.intensification.successive_halving import SuccessiveHalving
from smac.intensification.successive_halving_worker import SuccessiveHalvingWorker
from smac.runhistory import RunHistory, TrialInfo, TrialValue, TrialInfoIntent
from smac.runner.abstract_runner import StatusType
from smac.runner.target_algorithm_runner import TargetAlgorithmRunner
from smac.stats import Stats


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def evaluate_challenger(
    trial_info: TrialInfo,
    target_algorithm: TargetAlgorithmRunner,
    stats: Stats,
    runhistory: RunHistory,
    force_update=False,
):
    """
    Wrapper over challenger evaluation

    SMBO objects handles run history now, but to keep
    same testing functionality this function is a small
    wrapper to launch the target_algorithm and add it to the history
    """
    # evaluating configuration
    trial_info, result = target_algorithm.run_wrapper(trial_info=trial_info)
    stats._target_algorithm_walltime_used += float(result.time)
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
    value_from_config = trial_info.config.get_dictionary()["a"]

    return TrialValue(
        cost=value_from_config,
        time=0.5,
        status=StatusType.SUCCESS,
        starttime=time.time(),
        endtime=time.time() + 1,
        additional_info={},
    )


@pytest.fixture
def HB(make_scenario, make_stats, configspace_small) -> Hyperband:
    scenario = make_scenario(
        configspace_small,
        use_instances=True,
        n_instances=5,
        deterministic=False,
        min_budget=2,
        max_budget=5,
    )
    stats = make_stats(scenario)
    intensifier = Hyperband(scenario=scenario, eta=2, n_seeds=2)
    intensifier._stats = stats

    return intensifier


@pytest.fixture
def _HB(HB):
    return HyperbandWorker(HB)


@pytest.fixture
def make_hb_worker(make_scenario, make_stats, configspace_small):
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
        hb = Hyperband(
            scenario=scenario,
            instance_order=instance_order,
            incumbent_selection=incumbent_selection,
            min_challenger=min_challenger,
            eta=eta,
            n_seeds=n_seeds,
        )
        hb._stats = stats

        return HyperbandWorker(hb)

    return _make


@pytest.fixture
def make_target_algorithm():
    def _make(scenario, stats, func, required_arguments=[]):
        return TargetAlgorithmRunner(
            target_algorithm=func,
            scenario=scenario,
            stats=stats,
            required_arguments=required_arguments,
        )

    return _make


@pytest.fixture
def configs(configspace_small):
    configs = configspace_small.sample_configuration(20)
    return (configs[16], configs[15], configs[2], configs[3])


def test_initialization(runhistory: RunHistory, HB: Hyperband):
    """Makes sure that a proper_HB is created"""

    # We initialize the HB with zero intensifier_instances
    assert len(HB._intensifier_instances) == 0

    # Add an instance to check the_HB initialization
    assert HB._add_new_instance(n_workers=1)

    # Some default init
    assert HB._intensifier_instances[0]._hb_iters == 0
    assert HB._max_budget == 5
    assert HB._min_budget == 2

    # Update the stage
    HB._intensifier_instances[0]._update_stage(runhistory)

    # Parameters properly passed to SH
    assert len(HB._instance_seed_pairs) == 10
    assert HB._intensifier_instances[0]._successive_halving._min_budget == 2
    assert HB._intensifier_instances[0]._successive_halving._max_budget == 5


def test_process_results_via_sourceid(runhistory, HB: Hyperband, configs):
    """Makes sure source id is honored when deciding
    which_HB instance will consume the result/trial_info

    """
    # Mock the_HB instance so we can make sure the correct item is passed
    for i in range(10):
        HB._intensifier_instances[i] = mock.Mock()
        HB._intensifier_instances[i].process_results.return_value = (configs[0], 0.5)
        # make iter false so the mock object is not overwritten
        HB._intensifier_instances[i]._iteration_done = False

    # randomly create trial_infos and push into HB. Then we will make
    # sure they got properly allocated
    for i in np.random.choice(list(range(10)), 30):
        trial_info = TrialInfo(
            config=configs[0],
            instance=0,
            seed=0,
            budget=0.0,
            source=i,
        )

        # make sure results aren't messed up via magic variable
        # That is we check only the proper_HB instance has this
        magic = time.time()

        result = TrialValue(
            cost=1,
            time=0.5,
            status=StatusType.SUCCESS,
            starttime=1,
            endtime=2,
            additional_info=magic,
        )
        HB.process_results(
            trial_info=trial_info,
            trial_value=result,
            incumbent=None,
            runhistory=runhistory,
            time_bound=None,
            log_trajectory=False,
        )

        # Check the call arguments of each sh instance and make sure
        # it is the correct one

        # First the expected one
        assert HB._intensifier_instances[i].process_results.call_args[1]["trial_info"] == trial_info
        assert HB._intensifier_instances[i].process_results.call_args[1]["trial_value"] == result
        all_other_trial_infos, all_other_results = [], []
        for j, item in enumerate(HB._intensifier_instances):
            # Skip the expected_HB instance
            if i == j:
                continue
            if HB._intensifier_instances[j].process_results.call_args is None:
                all_other_trial_infos.append(None)
            else:
                all_other_trial_infos.append(HB._intensifier_instances[j].process_results.call_args[1]["trial_info"])
                all_other_results.append(HB._intensifier_instances[j].process_results.call_args[1]["trial_value"])

        assert trial_info not in all_other_trial_infos
        assert result not in all_other_results


def test_get_next_run_single_HB_instance(runhistory, HB: Hyperband, configs):
    """Makes sure that a single_HB instance returns a valid config"""

    challengers = configs[:4]
    for i in range(30):
        intent, trial_info = HB.get_next_run(
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

        # Add the config to rh in order to make HB aware that this
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

    # smax==1 (int(np.floor(np.log(max_budget / min_budget) / np.log(   eta))))
    assert HB._intensifier_instances[0]._s_max == 1

    # And we do not even complete 1 iteration, so s has to be 1
    assert HB._intensifier_instances[0]._s == 1

    # We should not create more_HB instance intensifier_instances
    assert len(HB._intensifier_instances) == 1

    # We are running with:
    # 'all_budgets': array([2.5, 5. ]) -> 2 intensifier_instances per config top
    # 'n_configs_in_stage': [2.0, 1.0],
    # This means we run int(2.5) + 2.0 = 4 runs before waiting
    assert i == 4

    # Also, check the internals of the unique sh instance

    # sh_min_budget==2.5 (eta ** -s * max_budget)
    assert HB._min_budget == 2

    # n_challengers=2 (int(np.floor((s_max + 1) / (s + 1)) * eta ** s))
    assert len(HB._intensifier_instances[0]._sh_intensifier._n_configs_in_stage) == 2


def test_get_next_run_multiple_HB_instances(runhistory, HB: Hyperband, configs):
    """Makes sure that two _HB instance can properly coexist and tag
    trial_info properly"""

    # We allow 2_HB instance to be created. This means, we have a newer iteration
    # to expect in hyperband
    challengers = configs[:4]
    trial_infos = []
    for i in range(30):
        intent, trial_info = HB.get_next_run(
            challengers=challengers,
            incumbent=None,
            get_next_configurations=None,
            runhistory=runhistory,
            n_workers=2,
        )
        trial_infos.append(trial_info)

        # Regenerate challenger list
        challengers = [c for c in challengers if c != trial_info.config]

        # Add the config to rh in order to make HB aware that this
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

    # We have not completed an iteration
    assert HB._intensifier_instances[0]._hb_iters == 0

    # Because n workers is now 2, we expect 2 sh intensifier_instances
    assert len(HB._intensifier_instances) == 2

    # Each of the intensifier_instances should have s equal to 1
    # As no iteration has been completed
    assert HB._intensifier_instances[0]._s_max == 1
    assert HB._intensifier_instances[0]._s == 1
    assert HB._intensifier_instances[1]._s_max == 1
    assert HB._intensifier_instances[1]._s == 1

    # First let us check everything makes sense in_HB-SH-0 HB-SH-0
    assert len(HB._intensifier_instances[0]._sh_intensifier._n_configs_in_stage) == 2
    assert len(HB._intensifier_instances[1]._sh_intensifier._n_configs_in_stage) == 2

    # We are running with:
    # + 4 runs for sh instance 0 ('all_budgets': array([2.5, 5. ]), 'n_configs_in_stage': [2.0, 1.0])
    #   that is, for SH0 we run in stage==0 int(2.5) intensifier_instances * 2.0 configs
    # And this times 2 because we have 2_HB intensifier_instances
    assert i == 8

    # Adding a new worker is not possible as we already have 2 intensifier_instances
    # and n_workers==2
    intent, trial_info = HB.get_next_run(
        challengers=challengers,
        incumbent=None,
        get_next_configurations=None,
        runhistory=runhistory,
        n_workers=2,
    )
    assert intent == TrialInfoIntent.WAIT


def test_add_new_instance(HB):
    """Test whether we can add a instance and when we should not"""

    # By default we do not create a_HB
    # test adding the first instance!
    assert len(HB._intensifier_instances) == 0
    assert HB._add_new_instance(n_workers=1)
    assert len(HB._intensifier_instances) == 1
    assert isinstance(HB._intensifier_instances[0], HyperbandWorker)
    # A second call should not add a new_HB instance
    assert not HB._add_new_instance(n_workers=1)

    # We try with 2_HB instance active

    # We effectively return true because we added a new_HB instance
    assert HB._add_new_instance(n_workers=2)

    assert len(HB._intensifier_instances) == 2
    assert isinstance(HB._intensifier_instances[1], HyperbandWorker)

    # Trying to add a third one should return false
    assert not HB._add_new_instance(n_workers=2)
    assert len(HB._intensifier_instances) == 2


def _exhaust_run_and_get_incumbent(runhistory, sh: SuccessiveHalving, configs, n_workers=1):
    """
    Runs all provided configs on all intensifier_instances and return the incumbent
    as a nice side effect runhistory/stats are properly filled
    """
    challengers = configs[:4]
    incumbent = None
    for i in range(100):
        try:
            intent, trial_info = sh.get_next_run(
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

        result = target_from_trial_info(trial_info)
        runhistory.add(
            config=trial_info.config,
            cost=result.cost,
            time=result.time,
            status=result.status,
            instance=trial_info.instance,
            seed=trial_info.seed,
            budget=trial_info.budget,
        )
        incumbent, inc_perf = sh.process_results(
            trial_info=trial_info,
            trial_value=result,
            incumbent=incumbent,
            runhistory=runhistory,
            time_bound=100.0,
            log_trajectory=False,
        )

    return incumbent, inc_perf


def test_parallel_same_as_serial_HB(make_hb_worker, configs):
    """Makes sure we behave the same as a serial run at the end"""
    runhistory1 = RunHistory()
    _HB = make_hb_worker(min_budget=2, max_budget=5, eta=2, n_instances=5)

    incumbent, inc_perf = _exhaust_run_and_get_incumbent(runhistory1, _HB, configs, n_workers=1)

    # Just to make sure nothing has changed from the_HB instance side to make
    # this check invalid:
    # We add config values, so config 3 with 0 and 7 should be the lesser cost
    assert incumbent == configs[0]
    assert inc_perf == 203

    # Do the same for HB, but have multiple_HB instance in there
    # This_HB instance will be created via n_workers==2
    # in _exhaust_run_and_get_incumbent
    runhistory2 = RunHistory()
    HB = make_hb_worker(min_budget=2, max_budget=5, eta=2, n_instances=5)._hyperband

    incumbent_phb, inc_perf_phb = _exhaust_run_and_get_incumbent(runhistory2, HB, configs)
    assert incumbent, incumbent_phb

    # This makes sure there is a single incumbent in HB
    assert inc_perf == inc_perf_phb

    # We don't want to loose any configuration, and particularly
    # we want to make sure the values of_HB instance to HB match
    assert len(runhistory1._data) == len(runhistory2._data)

    # Because it is a deterministic run, the run histories must be the
    # same on exhaustion
    assert runhistory1._data == runhistory2._data


def test_update_stage(make_hb_worker):
    """
    test initialization of all parameters and tracking variables
    """

    intensifier: HyperbandWorker = make_hb_worker(
        deterministic=True, n_instances=1, min_budget=0.1, max_budget=1, eta=2
    )

    # intensifier._update_stage()

    assert intensifier._s == 3
    assert intensifier._s_max == 3
    assert intensifier._hb_iters == 0
    assert isinstance(intensifier._sh_intensifier, SuccessiveHalvingWorker)
    assert intensifier._sh_intensifier._min_budget == 0.125
    assert intensifier._sh_intensifier._n_configs_in_stage == [8.0, 4.0, 2.0, 1.0]

    # next HB stage
    intensifier._update_stage()

    assert intensifier._s == 2
    assert intensifier._hb_iters == 0
    assert intensifier._sh_intensifier._min_budget == 0.25
    assert intensifier._sh_intensifier._n_configs_in_stage == [4.0, 2.0, 1.0]

    intensifier._update_stage()  # s = 1
    intensifier._update_stage()  # s = 0
    # HB iteration completed
    intensifier._update_stage()

    assert intensifier._s == intensifier._s_max
    assert intensifier._hb_iters == 1
    assert intensifier._sh_intensifier._min_budget == 0.125
    assert intensifier._sh_intensifier._n_configs_in_stage == [8.0, 4.0, 2.0, 1.0]


def test_eval_challenger(runhistory, make_target_algorithm, make_hb_worker, configs):
    """
    since hyperband uses eval_challenger and get_next_run of the internal successive halving,
    we don't test these method extensively
    """

    def target(x, seed, budget):
        return 0.1

    config1 = configs[0]
    config2 = configs[1]
    config3 = configs[2]

    intensifier = make_hb_worker(
        min_budget=0.5,
        max_budget=1,
        eta=2,
        n_instances=0,
        deterministic=True,
    )

    target_algorithm = make_target_algorithm(
        intensifier._scenario, intensifier._stats, target, required_arguments=["seed", "budget"]
    )

    # Testing get_next_run - get next configuration
    intent, trial_info = intensifier.get_next_run(
        challengers=[config2, config3],
        get_next_configurations=None,
        incumbent=None,
        runhistory=runhistory,
    )
    assert intensifier._s == intensifier._s_max
    assert trial_info.config == config2

    # Update to the last SH iteration of the given HB stage
    assert intensifier._s == 1
    assert intensifier._s_max == 1

    # We assume now that process results was called with below successes.
    # We track closely run execution through run_tracker, so this also
    # has to be update -- the fact that the succesive halving inside hyperband
    # processed the given configurations
    runhistory.add(config=config1, cost=1, time=1, status=StatusType.SUCCESS, seed=0, budget=1)
    intensifier._sh_intensifier._run_tracker[(config1, None, 0, 1)] = True
    runhistory.add(config=config2, cost=2, time=2, status=StatusType.SUCCESS, seed=0, budget=0.5)
    intensifier._sh_intensifier._run_tracker[(config2, None, 0, 0.5)] = True
    runhistory.add(config=config3, cost=3, time=2, status=StatusType.SUCCESS, seed=0, budget=0.5)
    intensifier._sh_intensifier._run_tracker[(config3, None, 0, 0.5)] = True

    intensifier._sh_intensifier._success_challengers = {config2, config3}
    intensifier._sh_intensifier._update_stage(runhistory)
    intent, trial_info = intensifier.get_next_run(
        challengers=[config2, config3],
        get_next_configurations=None,
        incumbent=None,
        runhistory=runhistory,
    )

    # evaluation should change the incumbent to config2
    assert trial_info.config is not None
    trial_value = evaluate_challenger(trial_info, target_algorithm, intensifier._stats, runhistory)

    inc, inc_value = intensifier.process_results(
        trial_info=trial_info,
        trial_value=trial_value,
        incumbent=config1,
        runhistory=runhistory,
        time_bound=np.inf,
    )

    assert inc == config2
    assert intensifier._s == 0
    assert inc_value == 0.1
    assert list(runhistory._data.keys())[-1].config_id, runhistory.config_ids[config2]
    assert intensifier._stats.incumbent_changed == 1


def test_budget_initialization(make_hb_worker):
    """
    Check computing budgets (only for non-instance cases)
    """

    intensifier = make_hb_worker(
        min_budget=1,
        max_budget=81,
        eta=3,
        n_instances=0,
        deterministic=True,
    )

    assert [1, 3, 9, 27, 81] == intensifier._all_budgets.tolist()
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
            [1.371742, 4.115226, 12.345679, 37.037037, 111.111111, 333.333333, 1000.0],
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

        intensifier = make_hb_worker(
            min_budget=minb,
            max_budget=maxb,
            eta=eta,
            n_instances=0,
            deterministic=True,
        )

        for i in range(len(all_budgets) + 10):
            comp_budgets = intensifier._sh_intensifier._all_budgets.tolist()
            comp_configs = intensifier._sh_intensifier._n_configs_in_stage

            assert isinstance(comp_configs, list)
            for c in comp_configs:
                assert isinstance(c, int)

            # all_budgets for SH is always a subset of all_budgets of HB
            np.testing.assert_array_almost_equal(all_budgets[i % len(all_budgets) :], comp_budgets, decimal=5)

            # The content of these lists might differ
            assert len(n_configs_in_stage[i % len(n_configs_in_stage) :]) == len(comp_configs)
            intensifier._update_stage()
