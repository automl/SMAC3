#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import sys
import os
import inspect

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from smac.runhistory.runhistory import average_cost
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    req_opts = parser.add_argument_group("Required Options")
    req_opts.add_argument("--scenario", required=True,
                          help="path to SMAC scenario")
    req_opts.add_argument("--trajectory", required=True,
                          help="path to SMAC trajectory")
    req_opts.add_argument("--output", required=True,
                          help="path to save runhistory to")

    req_opts = parser.add_argument_group("Optional Options")
    req_opts.add_argument("--configs", default="def+inc", type=str,
                          choices=["def", "inc", "def+inc", "wallclock_time",
                                   "cpu_time", "all"],
                          help="what configurations to evaluate. "
                               "def=default; inc=incumbent; "
                               "all=all configurations in the trajectory; "
                               "wallclock_time/cpu_time=evaluates at cpu- or "
                               "wallclock-timesteps of: [max_time/2^0, "
                               "max_time/2^1, max_time/2^3, ..., default] "
                               "with max_time being the highest recorded time")
    req_opts.add_argument("--instances", default="test", type=str,
                          choices=["train", "test", "train+test"],
                          help="what instances to evaluate")
    req_opts.add_argument('--epm', dest='epm', action='store_true',
                          help="Use EPM to validate")
    req_opts.add_argument('--no-epm', dest='epm', action='store_false',
                          help="Don't use EPM to validate")
    req_opts.set_defaults(epm=False)
    req_opts.add_argument("--runhistory", default=None, type=str, nargs='*',
                          help="path to one or more runhistories to take runs "
                               "from to either avoid recalculation or to train"
                               " the epm")
    req_opts.add_argument("--seed", type=int, help="random seed")
    req_opts.add_argument("--repetitions", default=1, type=int,
                          help="number of repetitions for nondeterministic "
                               "algorithms")
    req_opts.add_argument("--n_jobs", default=1, type=int,
                          help="number of cpu-cores to use (-1 to use all)")
    req_opts.add_argument("--tae", default="old", type=str,
                          help="what tae to use (if not using epm)", choices=["aclib", "old"])
    req_opts.add_argument("--verbose_level", default="INFO",
                          choices=["INFO", "DEBUG"],
                          help="verbose level")

    args_, misc = parser.parse_known_args()

    # remove leading '-' in option names
    misc = dict((k.lstrip("-"), v.strip("'"))
                for k, v in zip(misc[::2], misc[1::2]))

    if args_.verbose_level == "INFO":
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    scenario = Scenario(args_.scenario, cmd_options={'output_dir': ""})
    traj_logger = TrajLogger(None, Stats(scenario))
    trajectory = traj_logger.read_traj_aclib_format(args_.trajectory, scenario.cs)
    if args_.tae == "old":
        tae = ExecuteTARunOld(ta=scenario.ta,
                              run_obj=scenario.run_obj,
                              par_factor=scenario.par_factor,
                              cost_for_crash=scenario.cost_for_crash)
    if args_.tae == "aclib":
        tae = ExecuteTARunAClib(ta=scenario.ta,
                                run_obj=scenario.run_obj,
                                par_factor=scenario.par_factor,
                                cost_for_crash=scenario.cost_for_crash)

    validator = Validator(scenario, trajectory, args_.seed)

    # Load runhistory
    if args_.runhistory:
        runhistory = RunHistory(average_cost)
        for rh_path in args_.runhistory:
            runhistory.update_from_json(rh_path, scenario.cs)
    else:
        runhistory = None

    if args_.epm:
        validator.validate_epm(config_mode=args_.configs,
                               instance_mode=args_.instances,
                               repetitions=args_.repetitions,
                               runhistory=runhistory, output_fn=args_.output)
    else:
        validator.validate(config_mode=args_.configs,
                           instance_mode=args_.instances,
                           repetitions=args_.repetitions,
                           n_jobs=args_.n_jobs,
                           runhistory=runhistory,
                           tae=tae, output_fn=args_.output)
