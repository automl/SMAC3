import os

from collections import namedtuple
from joblib import Parallel, delayed
from typing import Union
import typing
import logging
import numpy as np

from smac.configspace import Configuration
from ConfigSpace.util import impute_inactive_values
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory, RunKey, StatusType
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import ExecuteTARun
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.utils.constants import MAXINT
from smac.utils.util_funcs import get_types

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

Run = namedtuple('Run', 'config inst seed inst_specs')

class Validator(object):
    """
    Validator for the output of SMAC-scenarios.
    evaluates specified configurations on specified instances.
    """

    def __init__(self, scenario: Scenario, trajectory: list, output: str,
                 rng: Union[np.random.RandomState, int]=None):
        """
        Construct Validator for given scenario and trajectory.

        Parameters
        ----------
        scenario: Scenario
            scenario object for cutoff, instances, features and specifics
        trajectory: trajectory-list
            trajectory to take incumbent(s) from
        output: string
            path to runhistory to be saved
        rng: np.random.RandomState or int
            Random number generator or seed
        """
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.scen = scenario
        self.traj = trajectory

        self.output = output

        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        elif isinstance(rng, int):
            self.rng = np.random.RandomState(seed=rng)
        else:
            num_run = np.random.randint(MAXINT)
            self.rng = np.random.RandomState(seed=num_run)

        self.rh = RunHistory(average_cost)  # update this rh with validation-runs

    def _save_results(self, rh:RunHistory, filename:str):
        """ Helper to save results to file """
        if self.output == "":
            self.logger.info("No output specified, validated runhistory not saved.")
        # Check if a folder or a file is specified as output
        if not self.output.endswith('.json'):
            old = self.output
            self.output = os.path.join(self.output, filename)
            self.logger.debug("Output is \"%s\", changing to \"%s\"!", old,
                              self.output)
        base = os.path.split(self.output)[0]
        if not base == "" and not os.path.exists(base):
            self.logger.debug("Folder (\"%s\") doesn't exist, creating.", base)
            os.makedirs(base)
        self.logger.info("Saving validation-results in %s", self.output)
        rh.save_json(self.output)
        return rh

    def validate(self, config_mode:Union[str, typing.List[Configuration]]='def',
                 instance_mode:Union[str, typing.List[str]]='test',
                 repetitions:int=1, n_jobs:int=1, backend:str='threading',
                 runhistory:RunHistory=None, tae:ExecuteTARun=None):
        """
        Validate configs on instances and save result in runhistory.

        Parameters
        ----------
        config_mode: str or list<Configuration>
            what configurations to validate
            either from [def, inc, def+inc, time, all], time means evaluation at
            timesteps 2^-4, 2^-3, 2^-2, 2^-1, 2^0, 2^1, ...
            or directly a list of Configuration
        instance_mode: str or list<str>
            what instances to use for validation, either from
            [train, test, train+test] or directly a list of instances
        repetitions: int
            number of repetitions in nondeterministic algorithms
        n_jobs: int
            number of parallel processes used by joblib
        backend: str
            what backend joblib should use for parallel runs
        runhistory: RunHistory
            optional, RunHistory-object to reuse runs
        tae: ExecuteTARun
            tae to be used. if None, will initialize ExecuteTARunOld

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


        # Get all runs to be evaluated as list
        runs = self.get_runs(config_mode, instance_mode, repetitions=repetitions,
                             runhistory=runhistory)

        # Create new Stats without limits
        inf_scen = Scenario({'run_obj':self.scen.run_obj,
                             'cutoff_time':self.scen.cutoff, 'output_dir':""})
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
            self.rh.add(config=runs[idx].config,
                        cost=result[1],
                        time=result[2],
                        status=result[0],
                        instance_id=runs[idx].inst,
                        seed=runs[idx].seed,
                        additional_info=result[3])
            idx += 1

        self._save_results(self.rh, 'validated_runhistory.json')
        return self.rh

    def _validate_parallel(self, tae: ExecuteTARun, runs: typing.List[Run],
                           n_jobs:int, backend:str):
        """
        Validate runs with joblibs Parallel-interface

        Parameters
        ----------
        tae: ExecuteTARun
            tae to be used for validation
        runs: list<Run>
            list with Run-objects
            [Run(config=CONFIG1,inst=INSTANCE1,seed=SEED1,inst_specs=INST_SPECIFICS1), ...]
        n_jobs: int
            number of cpus to use for validation (-1 to use all)
        backend: string
            what backend to use for parallelization

        Returns
        -------
        run_results: list<tuple(tae-returns)>
            results as returned by tae
        """
        # Runs with parallel
        run_results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_unbound_tae_starter)(tae, run.config,
                                          run.inst,
                                          self.scen.cutoff, run.seed,
                                          run.inst_specs,
                                          capped=False) for run in runs)
        return run_results

    def validate_epm(self, config_mode:Union[str, typing.List[Configuration]]='def',
                     instance_mode:Union[str, typing.List[str]]='test',
                     repetitions:int=1, runhistory:RunHistory=None) -> RunHistory:
        """
        Use EPM to predict costs/runtimes for unknown config/inst-pairs.

        Parameters
        ----------
        config_mode: str or list<Configuration>
            what configurations to validate
            either from [def, inc, def+inc, time, all], time means evaluation at
            timesteps 2^-4, 2^-3, 2^-2, 2^-1, 2^0, 2^1, ...
            or directly a list of Configuration
        instance_mode: str or list<str>
            what instances to use for validation, either from
            [train, test, train+test] or directly a list of instances
        repetitions: int
            number of repetitions in nondeterministic algorithms
        runhistory: RunHistory
            optional, RunHistory-object to reuse runs

        Returns
        -------
        runhistory: RunHistory
            runhistory with predicted runs
        """
        # Train random forest and transform training data (from given rh)
        # TODO: impute? how?
        rh2epm = RunHistory2EPM4Cost(num_params=len(self.scen.cs.get_hyperparameters()),
                                     scenario=self.scen, rng=self.rng)
        X, y = rh2epm.transform(runhistory)
        self.logger.debug("Training model with data of shape X: %s, y:%s",
                          str(X.shape), str(y.shape))

        types, bounds = get_types(self.scen.cs, self.scen.feature_array)
        model = RandomForestWithInstances(types=types,
                                          bounds=bounds,
                                          instance_features=self.scen.feature_array,
                                          seed=self.rng.randint(MAXINT),
                                          ratio_features=1.0)
        model.train(X, y)

        # Predict desired runs
        runs = self.get_runs(config_mode, instance_mode, repetitions, runhistory)
        try:
            feature_array_size = len(self.scen.cs.get_hyperparameters()) + self.scen.feature_array.shape[1]
        except AttributeError:
            feature_array_size = len(self.scen.cs.get_hyperparameters())
        X_pred = np.empty((len(runs), feature_array_size))
        for idx, run in enumerate(runs):
            try:
                X_pred[idx] = np.hstack([run.config.get_array(),
                                         self.scen.feature_dict[run.inst]])
            except AttributeError:
                X_pred[idx] = run.config.get_array()
        self.logger.debug("Predicting desired %d runs, data has shape %s and "
                          "NaNs are present: %s (if so, will be zeroed)",
                          len(runs), str(X_pred.shape), str(np.isnan(X_pred).any()))

        X_pred = np.nan_to_num(X_pred)  # Otherwise segfault in rfr
        y_pred = model.predict(X_pred)

        # Add runs to runhistory
        self.rh_epm = RunHistory(average_cost)
        self.rh_epm.update(runhistory)
        for run, pred in zip(runs, y_pred[0]):
            self.rh_epm.add(config=run.config,
                            cost=float(pred),
                            time=float(pred),
                            status=StatusType.SUCCESS,
                            instance_id=run.inst,
                            seed=-1,
                            additional_info={"additional_info":
                                "ESTIMATED USING EPM!"})

        self._save_results(self.rh_epm, 'validated_runhistory_EPM.json')
        return self.rh_epm

    def get_runs(self, configs: Union[str, typing.List[Configuration]],
                 insts: Union[str, typing.List[str]], repetitions: int=1,
                 runhistory: RunHistory=None) -> typing.List[Run]:
        """
        Generate list of SMAC-TAE runs to be executed. This means
        combinations of configs with all instances on a certain number of seeds.

        SideEffect: Adds runs that don't need to be reevaluated to self.rh!

        Parameters
        ----------
        configs: str or list<Configuration>
            what configurations to validate
            either from [def, inc, def+inc, time, all], time means evaluation at
            timesteps 2^-4, 2^-3, 2^-2, 2^-1, 2^0, 2^1, ...
            or directly a list of Configuration
        insts: str or list<str>
            what instances to use for validation, either from
            [train, test, train+test] or directly a list of instances
        repetitions: int
            number of seeds per instance/config-pair to be evaluated
        runhistory: RunHistory
            optional, try to reuse this runhistory and save some runs

        Returns
        -------
        runs: list<Run>
            list with Runs
            [Run(config=CONFIG1,inst=INSTANCE1,seed=SEED1,inst_specs=INST_SPECIFICS1),
             Run(config=CONFIG2,inst=INSTANCE2,seed=SEED2,inst_specs=INST_SPECIFICS2),
             ...]
        """
        # Get relevant configurations and instances
        if isinstance(configs, str):
            configs = self._get_configs(configs)
        if isinstance(insts, str):
            insts = self._get_instances(insts)

        # If no instances are given, fix the instances to one "None" instance
        if len(insts) == 0:
            insts = [None]
        # If algorithm is deterministic, fix repetitions to 1
        if self.scen.deterministic and repetitions != 1:
            self.logger.warning("Specified %d repetitions, but fixing to 1, "
                                "because algorithm is deterministic.", repetitions)
            repetitions = 1

        # Extract relevant information from given runhistory
        inst_seed_config = self._process_runhistory(configs, insts, runhistory)

        # Now create the actual run-list
        runs = []
        # Counter for runs without the need of recalculation
        runs_from_rh = 0

        for i in sorted(insts):
            for rep in range(repetitions):
                # First, find a seed and add all the data we can take from the
                # given runhistory to "our" validation runhistory.
                configs_evaluated = []
                if runhistory and i in inst_seed_config:
                    # Choose seed based on most often evaluated inst-seed-pair
                    seed, configs_evaluated = inst_seed_config[i].pop(0)
                    # Delete inst if all seeds are used
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
                # We now have a seed and add all configs that are not already
                # evaluated on that seed to the runs-list. This way, we
                # guarantee the same inst-seed-pairs for all configs.
                for config in [c for c in configs if not c in configs_evaluated]:
                    # Only use specifics if specific exists, else use string "0"
                    specs = self.scen.instance_specific[i] if i and i in self.scen.instance_specific else "0"
                    runs.append(Run(config=config,
                                    inst=i,
                                    seed=seed,
                                    inst_specs=specs))

        self.logger.info("Collected %d runs from %d configurations on %d "
                         "instances with %d repetitions. Reusing %d runs from "
                         "given runhistory.", len(runs), len(configs),
                         len(insts), repetitions, runs_from_rh)

        return runs

    def _process_runhistory(self, configs:typing.List[Configuration],
                            insts:typing.List[str], runhistory:RunHistory):
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


    def _get_configs(self, mode:str) -> typing.List[str]:
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

    def _get_instances(self, mode:str) -> typing.List[str]:
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
