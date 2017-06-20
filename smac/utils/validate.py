import numpy as np
import logging

from joblib import Parallel, delayed

from smac.utils.constants import MAXINT
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.stats.stats import Stats
from smac.optimizer.objective import average_cost

def _unbound_tae_starter(tae, *args, **kwargs):
    """
    Unbound function to be used by joblibs Parallel, since directly passing the
    TAE results in pickling-problems.

    Parameters
    ----------
    tae: ExecuteTARun
        tae to be used
    *args, **kwargs: various
        arguments to the tae

    Returns
    -------
    tae_results
    """
    return tae.start(*args, **kwargs)

def validate(scenario, trajectory, rng, output, config_mode='def',
             instances='TEST', repetitions=1, tae=None, runhistory=None, n_jobs=1):
    """
    Validate config on instances and save result in runhistory.

    Parameters
    ----------
    scenario: Scenario
        scenario object for cutoff, instances and specifics
    trajectory: Trajectory
        trajectory to take incumbent(s) from
    rng: np.random.RandomState
        Random number generator
    output: string
        path to runhistory to be saved
    config_mode: string
        what configurations to validate
        from [def, inc, def+inc, time], time means eval at 2^0, 2^1, 2^2, ...
    instances: string
        what instances to use for validation, from [train, test, train+test]
    tae: ExecuteTARun or None
        target algorithm executor to be used (including its runhistory)
    runhistory: RunHistory
        runhistory to be used (will be injected in tae)
    n_jobs: int
        number of parallel processes used by joblib

    Returns
    -------
    performance: float
        aggregated performance for configuration
    """
    # Add desired configs
    configs = []
    mode = config_mode.lower()
    if mode not in ['def', 'inc', 'def+inc', 'time']:
        raise ValueError("%s not a valid option for config_mode in validation."
                         %mode)
    if mode == "def" or mode == "def+inc":
        configs.append(scenario.cs.get_default_configuration())
    if mode == "inc" or mode == "def+inc":
        configs.append(trajectory[-1]["incumbent"])
    if mode == "time":
        configs.append(trajectory[0]["incumbent"])
        counter = 1
        while counter < len(trajectory):
            configs.append(trajectory[counter]["incumbent"])
            counter *= 2
        configs.append(trajectory[-1]["incumbent"])

    # Create new Stats without limits
    inf_scen = Scenario({'run_obj':'quality'})
    inf_stats = Stats(inf_scen)
    inf_stats.start_timing()

    # Create runhistory
    rh = RunHistory(average_cost)

    # Get all runs as list
    instances = instances.lower()
    use_train = instances == 'train' or instances == 'train+test'
    use_test = instances == 'test' or instances == 'train+test'
    runs = get_runs(configs, scenario, rng,
                    train=use_train, test=use_test, repetitions=repetitions)

    # Create TAE or inject stats
    if not tae:
        tae = ExecuteTARunOld(ta=scenario.ta,
                              stats=inf_stats,
                              run_obj=scenario.run_obj,
                              runhistory=rh,
                              par_factor=scenario.par_factor,
                              cost_for_crash=scenario.cost_for_crash)
    else:
        tae.stats = inf_stats

    # Runs with parallel
    Parallel(n_jobs=n_jobs)(delayed(sum)([i, i]) for i in range(100))
    run_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_unbound_tae_starter)(tae, run['config'],
                                      run['inst'],
                                      scenario.cutoff, run['seed'],
                                      run['inst_specs'],
                                      capped=False) for run in runs)

    # tae returns (status, cost, runtime, additional_info)
    # Add runs to RunHistory
    idx = 0
    for result in run_results:
        rh.add(config=runs[idx]['config'],
               cost=result[1],
               time=result[2],
               status=result[0],
               instance_id=runs[idx]['inst'],
               seed=runs[idx]['seed'],
               additional_info=result[3])
        idx += 1

    # add runs from passed runhistory
    if runhistory:
        rh.update(runhistory)

    # Save runhistory
    rh.save_json(output)

def get_runs(configs, scenario, rng, train=False, test=True, repetitions=1):
    """
    Generate list of dicts with SMAC-TAE runs to be executed. This means
    combinations of configs with all instances on a certain number of seeds.

    Parameters
    ----------
    configs: list<Configuration>
        configurations to be evaluated
    scenario: Scenario
        scenario object for cutoff, instances and specifics
    rng: np.random.RandomState
        Random number generator
    train: bool
        validate train-instances
    test: bool
        validate test-instances
    repetitions: int
        number of seeds per instance to be evaluated

    Returns
    -------
    runs: list<dict<string,string,string,string>>
        list with dicts (config/inst/seed/inst_specs)
    """
    if not train and not test:
        raise ValueError("No instances specified for validation.")

    runs = []

    # Get all instances
    insts = dict()
    if train:
        insts.update(scenario.train_insts)
    if test:
        insts.update(scenario.test_insts)

    for i in sorted(insts.keys()):
        for rep in range(repetitions):
            seed = rng.randint(MAXINT)
            # configs in inner loop -> same inst-seed-pairs for all configs
            for config in configs:
                runs.append({'config':config,
                             'inst':i,
                             'seed':seed,
                             'inst_specs': insts[i]})

    logging.info("Collected %d runs from %d configurations on %d instances "
                 "with %d repetitions.", len(runs), len(configs), len(insts),
                 repetitions)

    return runs

