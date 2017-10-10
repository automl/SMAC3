#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import sys
import os
import inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib
from smac.utils.validate import Validator
from smac.utils.io.traj_logging import TrajLogger


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
                          choices=["def", "inc", "def+inc", "time", "all"],
                          help="what configurations to evaluate: "
                               "def: default, inc: incumbent, "
                               "time: configs at timesteps 2^1, 2^2, 2^3, ..., "
                               "all: all configurations in the trajectory")
    req_opts.add_argument("--instances", default="test", type=str,
                          choices=["train", "test", "train+test"],
                          help="what instances to evaluate")
    req_opts.add_argument("--runhistory", default=None, type=str,
                          help="path to runhistory to impute runs from")
    req_opts.add_argument("--seed", type=int, help="random seed")
    req_opts.add_argument("--repetitions", default=1, type=int,
                          help="number of repetitions for nondeterministic "
                               "algorithms")
    req_opts.add_argument("--n_jobs", default=1, type=int,
                          help="number of cpu-cores to use")
    req_opts.add_argument("--tae", default="old", type=str,
                          help="what tae to use", choices=["aclib", "old"])
    req_opts.add_argument("--verbose_level", default=logging.INFO,
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

    scenario = Scenario(args_.scenario)
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

    validator = Validator(scenario, trajectory, args_.output,
                          args_.seed)
    validator.validate(config_mode=args_.configs,
                       instance_mode=args_.instances,
                       repetitions=args_.repetitions,
                       n_jobs=args_.n_jobs,
                       runhistory=args_.runhistory,
                       tae=tae)
