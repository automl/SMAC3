import os

import numpy as np
import logging
from joblib import Parallel, delayed

from smac.utils.constants import MAXINT
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory, RunKey
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_ta_run import ExecuteTARun
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
    tae_results: tuple
        return from tae.start
    """
    return tae.start(*args, **kwargs)

class Validator(object):
    """
    Validator for already run SMAC-scenarios, evaluates specified configurations
    on specified instances.
    """

    def __init__(self, scenario, trajectory, output, rng=None):
        """
        Construct Validator for given scenario and trajectory.

        Parameters
        ----------
        scenario: Scenario
            scenario object for cutoff, instances and specifics
        trajectory: Trajectory
            trajectory to take incumbent(s) from
        output: string
            path to runhistory to be saved
        rng: np.random.RandomState
            Random number generator
        """
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.scen = scenario
        self.traj = trajectory
        if output:
            self.output = output
        else:
            self.output =  "validation_rh.json"
        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        elif isinstance(rng, int):
            self.rng = np.random.RandomState(seed=rng)
        else:
            num_run = np.random.randint(MAXINT)
            self.rng = np.random.RandomState(seed=num_run)

        self.rh = RunHistory(average_cost)  # update this rh with validation-runs

    def validate(self, config_mode:str='def', instance_mode:str='test',
                 repetitions:int=1, n_jobs:int=1, backend:str='threading', runhistory:RunHistory=None,
                 tae:ExecuteTARun=None):
        """
        Validate configs on instances and save result in runhistory.

        Parameters
        ----------
        config_mode: string
            what configurations to validate
            from [def, inc, def+inc, time, all], time means evaluation at
            timesteps 2^-4, 2^-3, 2^-2, 2^-1, 2^0, 2^1, ...
        instance_mode: string
            what instances to use for validation, from [train, test, train+test]
        repetitions: int
            number of repetitions in nondeterministic algorithms
        n_jobs: int
            number of parallel processes used by joblib
        runhistory: RunHistory or string or None
            runhistory to take data from
        tae: ExecuteTARun
            tae to be used. if none, will initialize ExecuteTARunOld

        Returns
        -------
        runhistory: RunHistory
            runhistory with validated runs
        """
        self.logger.debug("Validating configs '%s' on instances '%s', repeating %d times"
                          " with %d parallel runs on backend '%s'.",
                          config_mode, instance_mode, repetitions, n_jobs, backend)
        # Reset runhistory
        self.rh = RunHistory(average_cost)

        # Get relevant configurations and instances
        configs = self._get_configs(config_mode)
        instances = self._get_instances(instance_mode)

        # If runhistory is given as string, load into memory
        if isinstance(runhistory, str):
            fn = runhistory
            runhistory = RunHistory(average_cost)
            runhistory.load_json(fn, self.scen.cs)

        # Get all runs needed as list
        runs = self.get_runs(configs, instances, repetitions=repetitions,
                             runhistory=runhistory)

        # Create new Stats without limits
        inf_scen = Scenario({'run_obj':self.scen.run_obj,
                             'cutoff_time':self.scen.cutoff, 'output_dir':None})
        inf_stats = Stats(inf_scen)
        inf_stats.start_timing()

        # Create TAE
        if not tae:
            tae = ExecuteTARunOld(ta=self.scen.ta,
                                  stats=inf_stats,
                                  run_obj=self.scen.run_obj,
                                  par_factor=self.scen.par_factor,
                                  cost_for_crash=self.scen.cost_for_crash)
        else:
            # Inject endless-stats
            tae.stats = inf_stats

        # Validate!
        run_results = self._validate_parallel(tae, runs, n_jobs, backend)

        # tae returns (status, cost, runtime, additional_info)
        # Add runs to RunHistory
        idx = 0
        for result in run_results:
            self.rh.add(config=runs[idx]['config'],
                        cost=result[1],
                        time=result[2],
                        status=result[0],
                        instance_id=runs[idx]['inst'],
                        seed=runs[idx]['seed'],
                        additional_info=result[3])
            idx += 1

        # Save runhistory
        if not self.output.endswith('.json'):
            old = self.output
            self.output = os.path.join(self.output, 'validated_runhistory.json')
            self.logger.debug("Output is \"%s\", changing to \"%s\"!", old,
                              self.output)
        base = os.path.split(self.output)[0]
        if not os.path.exists(base):
            self.logger.debug("Folder (\"%s\") doesn't exist, creating.", base)
            os.makedirs(base)
        self.logger.info("Saving validation-results in %s", self.output)
        self.rh.save_json(self.output)
        return self.rh

    def _validate_parallel(self, tae, runs, n_jobs, backend):
        """
        Validate runs with joblibs Parallel-interface

        Parameters
        ----------
        tae: ExecuteTARun
            tae to be used for validation
        runs: list<dict<string,string,string,string>>
            list with dicts
            [{"config":CONFIG,"inst":INSTANCE,"seed":SEED,"inst_specs":INST_SPECIFICS}]
        n_jobs: int
            number of cpus to use for validation
        backend: string
            what backend to use for parallelization

        Returns
        -------
        run_results: list<tuple(tae-returns)>
            results as returned by tae
        """
        # Runs with parallel
        run_results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_unbound_tae_starter)(tae, run['config'],
                                          run['inst'],
                                          self.scen.cutoff, run['seed'],
                                          run['inst_specs'],
                                          capped=False) for run in runs)
        return run_results

    def get_runs(self, configs, insts, repetitions=1, runhistory=None):
        """
        Generate list of SMAC-TAE runs to be executed. This means
        combinations of configs with all instances on a certain number of seeds.

        Parameters
        ----------
        configs: list<Configuration>
            configurations to be evaluated
        insts: list<strings>
            instances to be validated
        repetitions: int
            number of seeds per instance/config to be evaluated
        runhistory: RunHistory or None
            if given, try to reuse these results and save some runs

        Returns
        -------
        runs: list<dict<string,string,string,string>>
            list with dicts
            [{"config":CONFIG1,"inst":INSTANCE1,"seed":SEED1,"inst_specs":INST_SPECIFICS1},
             {"config":CONFIG2,"inst":INSTANCE2,"seed":SEED2,"inst_specs":INST_SPECIFICS2}]
        """
        # If no instances are given, fix the instances to one "None" instance
        if len(insts) == 0:
            insts = [None]
        # If algorithm is deterministic, fix repetitions to 1
        if self.scen.deterministic:
            self.logger.debug("Fixing repetitions to one, because algorithm is"
                              " deterministic.")
            repetitions = 1

        # Extract relevant information from given runhistory
        inst_seed_config = self._process_runhistory(configs, insts, runhistory)

        # Now create the actual run-list
        runs = []
        # Counter for runs without the need of recalculation
        runs_from_rh = 0

        for i in sorted(insts):
            for rep in range(repetitions):
                configs_evaluated = []
                if runhistory and i in inst_seed_config:
                    # Choose seed based on most often evaluated inst-seed-pair
                    seed, configs_evaluated = inst_seed_config[i].pop(0)
                    # Delete i from dict if list is empty
                    if len(inst_seed_config[i]) == 0:
                        inst_seed_config.pop(i)
                    # Add runs to runhistory
                    for c in configs_evaluated:
                        runkey = RunKey(runhistory.config_ids[c], i, seed)
                        cost, time, status, additional_info = runhistory.data[runkey]
                        self.rh.add(c, cost, time, status, instance_id=i,
                                    seed=seed, additional_info=additional_info)
                        runs_from_rh += 1
                else:
                    # If no runhistory or no entries for instance, get new seed
                    seed = self.rng.randint(MAXINT)
                    if self.scen.deterministic:
                        seed = 0
                # configs in inner loop -> same inst-seed-pairs for all configs
                for config in [c for c in configs if not c in
                               configs_evaluated]:
                    specs = self.scen.instance_specific[i] if i and i in self.scen.instance_specific else "0"
                    runs.append({'config':config,
                                 'inst':i,
                                 'seed':seed,
                                 'inst_specs': specs})

        self.logger.info("Collected %d runs from %d configurations on %d instances "
                         "with %d repetitions.", len(runs), len(configs), len(insts),
                         repetitions)
        self.logger.info("Using %d runs from given runhistory.", runs_from_rh)

        return runs

    def _process_runhistory(self, configs, insts, runhistory):
        """
        Processes runhistory from self.get_runs by extracting already evaluated
        (relevant) config-inst-seed tuples.

        Parameters
        ----------
        configs: list(Configuration)
            list of configs of interest
        insts: list(str)
            list of instances of interest
        runhistory: RunHistory
            runhistory to extract runs from

        Returns
        -------
        inst_seed_config: dict<str : list(tuple(int, tuple(configs)))>
            dictionary mapping instances to a list of tuples of already used
            seeds and the configs that this inst-seed-pair has been evaluated
            on, sorted by the number of configs
        """
        # We want to reuse seeds that have been used on most configurations
        # To this end, we create a dictionary as {instances:{seed:[configs]}}
        # Like this we can easily retrieve the most used instance-seed pairs to
        # minimize the number of runs to be evaluated
        inst_seed_config = {}
        if runhistory:
            relevant = dict()
            for key in runhistory.data:
                if (runhistory.ids_config[key.config_id] in configs
                        and key.instance_id in insts):
                    relevant[key] = runhistory.data[key]

            # Change data-structure to {instances:[(seed1, (configs)), (seed2, (configs), ... ]}
            # to make most used seed easily accessible, we sort after length of configs
            for key in relevant:
                inst, seed = key.instance_id, key.seed
                config = runhistory.ids_config[key.config_id]
                if inst in inst_seed_config:
                    if seed in inst_seed_config[inst]:
                        inst_seed_config[inst][seed].append(config)
                    else:
                        inst_seed_config[inst][seed] = [config]
                else:
                    inst_seed_config[inst] = {seed : [config]}

            inst_seed_config = {i :
                                sorted([(seed, tuple(inst_seed_config[i][seed]))
                                        for seed in inst_seed_config[i]],
                                       key=lambda x: len(x[1]))
                                for i in inst_seed_config}
        return inst_seed_config


    def _get_configs(self, mode):
        """
        Return desired configs

        Parameters
        ----------
        mode : string
            from [def, inc, def+inc, time, all], time means evaluation at
            timesteps 2^-4, 2^-3, 2^-2, 2^-1, 2^0, 2^1, ...

        Returns
        -------
        configs: list<Configuration>
            list with desired configurations
        """
        # Add desired configs
        configs = []
        mode = mode.lower()
        if mode not in ['def', 'inc', 'def+inc', 'time', 'all']:
            raise ValueError("%s not a valid option for config_mode in validation."
                             %mode)
        if mode == "def" or mode == "def+inc":
            configs.append(self.scen.cs.get_default_configuration())
        if mode == "inc" or mode == "def+inc":
            configs.append(self.traj[-1]["incumbent"])
        if mode == "time":
            # add configs at evaluations 2^1, 2^2, 2^3, ...
            configs.append(self.traj[0]["incumbent"])  # add first
            counter = 2^(-4)
            for entry in self.traj[:-1]:
                if (entry["wallclock_time"] >= counter and
                        entry["incumbent"] not in configs):
                    configs.append(entry["incumbent"])
                    counter *= 2
            configs.append(self.traj[-1]["incumbent"])  # add last
        if mode == "all":
            for entry in self.traj:
                if not entry["incumbent"] in configs:
                    configs.append(entry["incumbent"])
        self.logger.debug("Gathered %d configurations for mode %s.",
                len(configs), mode)
        return configs

    def _get_instances(self, mode):
        """
        Get desired instances

        Parameters
        ----------
        mode: string
            what instances to use for validation, from [train, test, train+test]

        Returns
        -------
        instances: list<strings>
            instances to be used
        """
        instance_mode = mode.lower()
        if mode not in ['train', 'test', 'train+test']:
            raise ValueError("%s not a valid option for instance_mode in validation."
                             %mode)

        # Make sure if instances matter, than instances should be passed
        if ((instance_mode == 'train' and self.scen.train_insts == [None]) or
            (instance_mode == 'test' and self.scen.test_insts == [None])):
            self.logger.warning("Instance mode is set to %s, but there are no "
            "%s-instances specified in the scenario. Setting instance mode to"
            "\"train+test\"!", instance_mode, instance_mode)
            instance_mode = 'train+test'

        instances = []
        if ((instance_mode == 'train' or instance_mode == 'train+test') and not
                self.scen.train_insts == [None]):
            instances.extend(self.scen.train_insts)
        if ((instance_mode == 'test' or instance_mode == 'train+test') and not
                self.scen.test_insts == [None]):
            instances.extend(self.scen.test_insts)
        return instances
