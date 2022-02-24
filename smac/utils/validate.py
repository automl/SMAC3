import os

from collections import namedtuple
from joblib import Parallel, delayed
from typing import Union
import typing
import logging
import numpy as np

from smac.configspace import Configuration, convert_configurations_to_array
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.rfr_imputator import RFRImputator
from smac.epm.util_funcs import get_types
from smac.runhistory.runhistory import RunHistory, RunInfo, RunKey, RunValue, StatusType
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.base import BaseRunner
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.utils.constants import MAXINT

__author__ = "Joshua Marben"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Joshua Marben"
__email__ = "joshua.marben@neptun.uni-freiburg.de"


def _unbound_tae_starter(
    tae: BaseRunner, runhistory: typing.Optional[RunHistory], run_info: RunInfo,
    *args: typing.Any, **kwargs: typing.Any
) -> RunValue:
    """
    Unbound function to be used by joblibs Parallel, since directly passing the
    TAE results in pickling-problems.

    Parameters
    ----------
    tae: BaseRunner
        tae to be used
    runhistory: RunHistory
        runhistory to save
    run_info: RunInfo
        Config to be launched
    *args, **kwargs: various
        arguments to the tae

    Returns
    -------
    tae_results: RunValue
        return from tae.start
    """
    run_info, result = tae.run_wrapper(run_info)
    tae.stats.submitted_ta_runs += 1
    tae.stats.finished_ta_runs += 1
    tae.stats.ta_time_used += float(result.time)
    if runhistory:
        runhistory.add(
            config=run_info.config,
            cost=result.cost,
            time=result.time,
            status=result.status,
            instance_id=run_info.instance,
            seed=run_info.seed,
            budget=run_info.budget,
        )
        tae.stats.n_configs = len(runhistory.config_ids)

    return result


_Run = namedtuple('Run', 'config inst seed inst_specs')


class Validator(object):
    """
    Validator for the output of SMAC-scenarios.

    Evaluates specified configurations on specified instances.

    Parameters
    ----------
    scenario: Scenario
        scenario object for cutoff, instances, features and specifics
    trajectory: trajectory-list
        trajectory to take incumbent(s) from
    rng: np.random.RandomState or int
        Random number generator or seed
    """

    def __init__(self,
                 scenario: Scenario,
                 trajectory: typing.Optional[typing.List],
                 rng: Union[np.random.RandomState, int, None] = None) -> None:
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.traj = trajectory
        self.scen = scenario
        self.epm = None  # type: typing.Optional[RandomForestWithInstances]

        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        elif isinstance(rng, int):
            self.rng = np.random.RandomState(seed=rng)
        else:
            self.logger.debug('no seed given, using default seed of 1')
            num_run = 1
            self.rng = np.random.RandomState(seed=num_run)

    def _save_results(
        self,
        rh: RunHistory,
        output_fn: typing.Optional[str],
        backup_fn: typing.Optional[str] = None,
    ) -> None:
        """ Helper to save results to file

        Parameters
        ----------
        rh: RunHistory
            runhistory to save
        output_fn: str
            if ends on '.json': filename to save history to
            else: directory to save runhistory to (filename is backup_fn)
        backup_fn: str
            if output_fn does not end on '.json', treat output_fn as dir and
            append backup_fn as filename (if output_fn ends on '.json', this
            argument is ignored)
        """
        if not output_fn:
            self.logger.info("No output specified, validated runhistory not saved.")
            return
        # Check if a folder or a file is specified as output
        if not output_fn.endswith('.json'):
            if backup_fn is None:
                raise ValueError('If output_fn does not end with .json the argument backup_fn needs to be given.')
            output_dir = output_fn
            output_fn = os.path.join(output_dir, backup_fn)
            self.logger.debug("Output is \"%s\", changing to \"%s\"!", output_dir, output_fn)
        base = os.path.split(output_fn)[0]
        if not base == "" and not os.path.exists(base):
            self.logger.debug("Folder (\"%s\") doesn't exist, creating.", base)
            os.makedirs(base)
        rh.save_json(output_fn)
        self.logger.info("Saving validation-results in %s", output_fn)

    def validate(self,
                 config_mode: Union[str, typing.List[Configuration]] = 'def',
                 instance_mode: Union[str, typing.List[str]] = 'test',
                 repetitions: int = 1,
                 n_jobs: int = 1,
                 backend: str = 'threading',
                 runhistory: typing.Optional[RunHistory] = None,
                 tae: BaseRunner = None,
                 output_fn: typing.Optional[str] = None,
                 ) -> RunHistory:
        """
        Validate configs on instances and save result in runhistory.
        If a runhistory is provided as input it is important that you run it on the same/comparable hardware.

        side effect: if output is specified, saves runhistory to specified
        output directory.

        Parameters
        ----------
        config_mode: str or list<Configuration>
            string or directly a list of Configuration.
            string from [def, inc, def+inc, wallclock_time, cpu_time, all].
            time evaluates at cpu- or wallclock-timesteps of:
            [max_time/2^0, max_time/2^1, max_time/2^3, ..., default]
            with max_time being the highest recorded time
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
        tae: BaseRunner
            tae to be used. if None, will initialize ExecuteTARunOld
        output_fn: str
            path to runhistory to be saved. if the suffix is not '.json', will
            be interpreted as directory and filename will be
            'validated_runhistory.json'

        Returns
        -------
        runhistory: RunHistory
            runhistory with validated runs
        """
        self.logger.debug("Validating configs '%s' on instances '%s', repeating %d times"
                          " with %d parallel runs on backend '%s'.",
                          config_mode, instance_mode, repetitions, n_jobs, backend)

        # Get all runs to be evaluated as list
        runs, validated_rh = self._get_runs(config_mode, instance_mode, repetitions, runhistory)

        # Create new Stats without limits
        inf_scen = Scenario({
            'run_obj': self.scen.run_obj,
            'cutoff_time': self.scen.cutoff,  # type: ignore[attr-defined] # noqa F821
            'output_dir': ""})
        inf_stats = Stats(inf_scen)
        inf_stats.start_timing()

        # Create TAE
        if not tae:
            tae = ExecuteTARunOld(ta=self.scen.ta,  # type: ignore[attr-defined] # noqa F821
                                  stats=inf_stats,
                                  run_obj=self.scen.run_obj,
                                  par_factor=self.scen.par_factor,  # type: ignore[attr-defined] # noqa F821
                                  cost_for_crash=self.scen.cost_for_crash)  # type: ignore[attr-defined] # noqa F821
        else:
            # Inject endless-stats
            tae.stats = inf_stats

        # Validate!
        run_results = self._validate_parallel(tae, runs, n_jobs, backend, runhistory)
        assert len(run_results) == len(runs), (run_results, runs)

        # tae returns (status, cost, runtime, additional_info)
        # Add runs to RunHistory
        for run, result in zip(runs, run_results):
            validated_rh.add(config=run.config,
                             cost=result.cost,
                             time=result.time,
                             status=result.status,
                             instance_id=run.inst,
                             seed=run.seed,
                             additional_info=result.additional_info)

        self._save_results(validated_rh, output_fn, backup_fn="validated_runhistory.json")
        return validated_rh

    def _validate_parallel(
        self,
        tae: BaseRunner,
        runs: typing.List[_Run],
        n_jobs: int,
        backend: str,
        runhistory: typing.Optional[RunHistory] = None,
    ) -> typing.List[RunValue]:
        """
        Validate runs with joblibs Parallel-interface

        Parameters
        ----------
        tae: BaseRunner
            tae to be used for validation
        runs: list<_Run>
            list with _Run-objects
            [_Run(config=CONFIG1,inst=INSTANCE1,seed=SEED1,inst_specs=INST_SPECIFICS1), ...]
        n_jobs: int
            number of cpus to use for validation (-1 to use all)
        backend: str
            what backend to use for parallelization
        runhistory: RunHistory
            optional, RunHistory-object to reuse runs

        Returns
        -------
        run_results: list<tuple(tae-returns)>
            results as returned by tae
        """
        # Runs with parallel
        run_results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_unbound_tae_starter)(
                tae,
                runhistory,
                RunInfo(
                    config=run.config,
                    instance=run.inst,
                    instance_specific="0",
                    seed=run.seed,
                    cutoff=self.scen.cutoff,  # type: ignore[attr-defined] # noqa F821
                    capped=False,
                    budget=0
                )
            ) for run in runs)
        return run_results

    def validate_epm(self,
                     config_mode: Union[str, typing.List[Configuration]] = 'def',
                     instance_mode: Union[str, typing.List[str]] = 'test',
                     repetitions: int = 1,
                     runhistory: typing.Optional[RunHistory] = None,
                     output_fn: typing.Optional[str] = None,
                     reuse_epm: bool = True,
                     ) -> RunHistory:
        """
        Use EPM to predict costs/runtimes for unknown config/inst-pairs.

        side effect: if output is specified, saves runhistory to specified
        output directory.

        Parameters
        ----------
        output_fn: str
            path to runhistory to be saved. if the suffix is not '.json', will
            be interpreted as directory and filename will be
            'validated_runhistory_EPM.json'
        config_mode: str or list<Configuration>
            string or directly a list of Configuration, string from [def, inc, def+inc, wallclock_time, cpu_time, all].
            time evaluates at cpu- or wallclock-timesteps of:
            [max_time/2^0, max_time/2^1, max_time/2^3, ..., default] with max_time being the highest recorded time
        instance_mode: str or list<str>
            what instances to use for validation, either from
            [train, test, train+test] or directly a list of instances
        repetitions: int
            number of repetitions in nondeterministic algorithms
        runhistory: RunHistory
            optional, RunHistory-object to reuse runs
        reuse_epm: bool
            if true (and if `self.epm`), reuse epm to validate runs

        Returns
        -------
        runhistory: RunHistory
            runhistory with predicted runs
        """
        if not isinstance(runhistory, RunHistory) and (self.epm is None or not reuse_epm):
            raise ValueError("No runhistory specified for validating with EPM!")
        elif not reuse_epm or self.epm is None:
            # Create RandomForest
            types, bounds = get_types(self.scen.cs, self.scen.feature_array)  # type: ignore[attr-defined] # noqa F821
            epm = RandomForestWithInstances(
                configspace=self.scen.cs,  # type: ignore[attr-defined] # noqa F821
                types=types,
                bounds=bounds,
                instance_features=self.scen.feature_array,
                seed=self.rng.randint(MAXINT),
                ratio_features=1.0,
            )
            # Use imputor if objective is runtime
            imputor = None
            impute_state = None
            impute_censored_data = False
            if self.scen.run_obj == 'runtime':
                threshold = self.scen.cutoff * self.scen.par_factor  # type: ignore[attr-defined] # noqa F821
                imputor = RFRImputator(rng=self.rng,
                                       cutoff=self.scen.cutoff,  # type: ignore[attr-defined] # noqa F821
                                       threshold=threshold,
                                       model=epm)
                impute_censored_data = True
                impute_state = [StatusType.CAPPED]
                success_states = [StatusType.SUCCESS, ]
            else:
                success_states = [StatusType.SUCCESS, StatusType.CRASHED, StatusType.MEMOUT]

            # Transform training data (from given rh)
            rh2epm = RunHistory2EPM4Cost(num_params=len(self.scen.cs.get_hyperparameters()),  # type: ignore[attr-defined] # noqa F821
                                         scenario=self.scen, rng=self.rng,
                                         impute_censored_data=impute_censored_data,
                                         imputor=imputor,
                                         impute_state=impute_state,
                                         success_states=success_states)
            assert runhistory is not None  # please mypy
            X, y = rh2epm.transform(runhistory)
            self.logger.debug("Training model with data of shape X: %s, y:%s",
                              str(X.shape), str(y.shape))
            # Train random forest
            epm.train(X, y)
        else:
            epm = typing.cast(RandomForestWithInstances, self.epm)

        # Predict desired runs
        runs, rh_epm = self._get_runs(config_mode, instance_mode, repetitions, runhistory)

        feature_array_size = len(self.scen.cs.get_hyperparameters())  # type: ignore[attr-defined] # noqa F821
        if self.scen.feature_array is not None:
            feature_array_size += self.scen.feature_array.shape[1]

        X_pred = np.empty((len(runs), feature_array_size))
        for idx, run in enumerate(runs):
            if self.scen.feature_array is not None and run.inst is not None:
                X_pred[idx] = np.hstack([convert_configurations_to_array([run.config])[0],
                                         self.scen.feature_dict[run.inst]])
            else:
                X_pred[idx] = convert_configurations_to_array([run.config])[0]
        self.logger.debug("Predicting desired %d runs, data has shape %s",
                          len(runs), str(X_pred.shape))

        y_pred = epm.predict(X_pred)
        self.epm = epm

        # Add runs to runhistory
        for run, pred in zip(runs, y_pred[0]):
            rh_epm.add(config=run.config,
                       cost=float(pred),
                       time=float(pred),
                       status=StatusType.SUCCESS,
                       instance_id=run.inst,
                       seed=-1,
                       additional_info={"additional_info":
                                        "ESTIMATED USING EPM!"})

        if output_fn:
            self._save_results(rh_epm, output_fn, backup_fn="validated_runhistory_EPM.json")
        return rh_epm

    def _get_runs(self,
                  configs: Union[str, typing.List[Configuration]],
                  insts: Union[str, typing.List[str]],
                  repetitions: int = 1,
                  runhistory: RunHistory = None,
                  ) -> typing.Tuple[typing.List[_Run], RunHistory]:
        """ Generate list of SMAC-TAE runs to be executed. This means
        combinations of configs with all instances on a certain number of seeds.

        side effect: Adds runs that don't need to be reevaluated to self.rh!

        Parameters
        ----------
        configs: str or list<Configuration>
            string or directly a list of Configuration
            str from [def, inc, def+inc, wallclock_time, cpu_time, all]
            time evaluates at cpu- or wallclock-timesteps of:
            [max_time/2^0, max_time/2^1, max_time/2^3, ..., default]
            with max_time being the highest recorded time
        insts: str or list<str>
            what instances to use for validation, either from
            [train, test, train+test] or directly a list of instances
        repetitions: int
            number of seeds per instance/config-pair to be evaluated
        runhistory: RunHistory
            optional, try to reuse this runhistory and save some runs

        Returns
        -------
        runs: list<_Run>
            list with _Runs
            [_Run(config=CONFIG1,inst=INSTANCE1,seed=SEED1,inst_specs=INST_SPECIFICS1),
             _Run(config=CONFIG2,inst=INSTANCE2,seed=SEED2,inst_specs=INST_SPECIFICS2),
             ...]
        """
        # Get relevant configurations and instances
        if isinstance(configs, str):
            configs = self._get_configs(configs)
        if isinstance(insts, str):
            instances = sorted(self._get_instances(insts))  # type: typing.Sequence[typing.Union[str, None]]
        elif insts is not None:
            instances = sorted(insts)
        else:
            instances = [None]
        # If no instances are given, fix the instances to one "None" instance
        if not instances:
            instances = [None]

        # If algorithm is deterministic, fix repetitions to 1
        if self.scen.deterministic and repetitions != 1:  # type: ignore[attr-defined] # noqa F821
            self.logger.warning("Specified %d repetitions, but fixing to 1, "
                                "because algorithm is deterministic.", repetitions)
            repetitions = 1

        # Extract relevant information from given runhistory
        inst_seed_config = self._process_runhistory(configs, instances, runhistory)

        # Now create the actual run-list
        runs = []
        # Counter for runs without the need of recalculation
        runs_from_rh = 0
        # If we reuse runs, we want to return them as well
        new_rh = RunHistory()

        for i in instances:
            for rep in range(repetitions):
                # First, find a seed and add all the data we can take from the
                # given runhistory to "our" validation runhistory.
                configs_evaluated = []  # type: Configuration
                if runhistory and i in inst_seed_config:
                    # Choose seed based on most often evaluated inst-seed-pair
                    seed, configs_evaluated = inst_seed_config[i].pop(0)
                    # Delete inst if all seeds are used
                    if not inst_seed_config[i]:
                        inst_seed_config.pop(i)
                    # Add runs to runhistory
                    for c in configs_evaluated[:]:
                        runkey = RunKey(runhistory.config_ids[c], i, seed)
                        cost, time, status, start, end, additional_info = runhistory.data[runkey]
                        if status in [StatusType.CRASHED, StatusType.ABORT, StatusType.CAPPED]:
                            # Not properly executed target algorithm runs should be repeated
                            configs_evaluated.remove(c)
                            continue
                        new_rh.add(c, cost, time, status, instance_id=i,
                                   seed=seed, starttime=start, endtime=end,
                                   additional_info=additional_info)
                        runs_from_rh += 1
                else:
                    # If no runhistory or no entries for instance, get new seed
                    seed = self.rng.randint(MAXINT)

                # We now have a seed and add all configs that are not already
                # evaluated on that seed to the runs-list. This way, we
                # guarantee the same inst-seed-pairs for all configs.
                for config in [c for c in configs if c not in configs_evaluated]:
                    # Only use specifics if specific exists, else use string "0"
                    specs = self.scen.instance_specific[i] if i and i in self.scen.instance_specific else "0"
                    runs.append(_Run(config=config,
                                     inst=i,
                                     seed=seed,
                                     inst_specs=specs))

        self.logger.info("Collected %d runs from %d configurations on %d "
                         "instances with %d repetitions. Reusing %d runs from "
                         "given runhistory.", len(runs), len(configs),
                         len(instances), repetitions, runs_from_rh)

        return runs, new_rh

    def _process_runhistory(
        self,
        configs: typing.List[Configuration],
        insts: typing.Sequence[typing.Optional[str]],
        runhistory: typing.Optional[RunHistory],
    ) -> typing.Dict[str, typing.List[typing.Tuple[int, typing.List[Configuration]]]]:
        """
        Processes runhistory from self._get_runs by extracting already evaluated
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
        if runhistory:
            inst_seed_config = {}  # type: typing.Dict[str, typing.Dict[int, typing.List[Configuration]]]
            relevant = dict()
            for key in runhistory.data:
                if (runhistory.ids_config[key.config_id] in configs and key.instance_id in insts):
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
                    inst_seed_config[inst] = {seed: [config]}

            return {
                i: sorted(
                    [(seed, list(inst_seed_config[i][seed])) for seed in inst_seed_config[i]],
                    key=lambda x: len(x[1])
                ) for i in inst_seed_config
            }
        else:
            rval = {}  # type: typing.Dict[str, typing.List[typing.Tuple[int, typing.List[Configuration]]]]
            return rval

    def _get_configs(self, mode: str) -> typing.List[str]:
        """
        Return desired configs

        Parameters
        ----------
        mode: str
            str from [def, inc, def+inc, wallclock_time, cpu_time, all]
                time evaluates at cpu- or wallclock-timesteps of:
                [max_time/2^0, max_time/2^1, max_time/2^3, ..., default]
                with max_time being the highest recorded time

        Returns
        -------
        configs: list<Configuration>
            list with desired configurations
        """

        # Get trajectory and make sure it's not None to please mypy
        traj = self.traj
        assert traj is not None  # please mypy

        # Add desired configs
        configs = []
        mode = mode.lower()
        if mode not in ['def', 'inc', 'def+inc', 'wallclock_time', 'cpu_time',
                        'all']:
            raise ValueError("%s not a valid option for config_mode in validation."
                             % mode)
        if mode == "def" or mode == "def+inc":
            configs.append(self.scen.cs.get_default_configuration())  # type: ignore[attr-defined] # noqa F821
        if mode == "inc" or mode == "def+inc":
            configs.append(traj[-1]["incumbent"])
        if mode in ["wallclock_time", "cpu_time"]:
            # get highest time-entry and add entries from there
            # not using wallclock_limit in case it's inf
            if (mode == "wallclock_time" and np.isfinite(self.scen.wallclock_limit)):
                max_time = self.scen.wallclock_limit
            elif (mode == "cpu_time" and np.isfinite(self.scen.algo_runs_timelimit)):
                max_time = self.scen.algo_runs_timelimit
            else:
                max_time = traj[-1][mode]
            counter = 2 ** 0
            for entry in traj[::-1]:
                if (entry[mode] <= max_time / counter and entry["incumbent"] not in configs):
                    configs.append(entry["incumbent"])
                    counter *= 2
            if not traj[0]["incumbent"] in configs:
                configs.append(traj[0]["incumbent"])  # add first
        if mode == "all":
            for entry in traj:
                if not entry["incumbent"] in configs:
                    configs.append(entry["incumbent"])
        self.logger.debug("Gathered %d configurations for mode %s.",
                          len(configs), mode)
        return configs

    def _get_instances(self, mode: str) -> typing.List[str]:
        """
        Get desired instances

        Parameters
        ----------
        mode: str
            what instances to use for validation, from [train, test, train+test]

        Returns
        -------
        instances: list<str>
            instances to be used
        """
        instance_mode = mode.lower()
        if mode not in ['train', 'test', 'train+test']:
            raise ValueError("%s not a valid option for instance_mode in validation."
                             % mode)

        # Make sure if instances matter, than instances should be passed
        if ((instance_mode == 'train' and self.scen.train_insts == [None]) or (
                instance_mode == 'test' and self.scen.test_insts == [None])):
            self.logger.warning("Instance mode is set to %s, but there are no "
                                "%s-instances specified in the scenario. Setting instance mode to"
                                "\"train+test\"!", instance_mode, instance_mode)
            instance_mode = 'train+test'

        instances = []  # type: typing.List[str]
        if (instance_mode == 'train' or instance_mode == 'train+test') and not self.scen.train_insts == [None]:
            instances.extend(self.scen.train_insts)
        if (instance_mode == 'test' or instance_mode == 'train+test') and not self.scen.test_insts == [None]:
            instances.extend(self.scen.test_insts)
        return instances
