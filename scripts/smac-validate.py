#!/usr/bin/env python

from argparse import ArgumentParser, RawTextHelpFormatter
import logging
import sys
import os
import inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib
from smac.utils.validate import Validator
from smac.utils.io.traj_logging import TrajLogger


if __name__ == "__main__":


    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
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
                          help="what configurations to evaluate:\n"
                               "  def: default\n  inc: incumbent\n"
                               "  time: configs at timesteps 2^1, 2^2, 2^3, ...\n"
                               "  all: all configurations in the trajectory")
    req_opts.add_argument("--instances", default="test", type=str,
                          choices=["train", "test", "train+test"],
                          help="what instances to evaluate")
    req_opts.add_argument("--use_epm", default=False,
                          help="whether to use an EPM instead of evaluating "
                               "all runs with the TAE")
    req_opts.add_argument("--runhistory", default=None, type=str,
                          help="path to runhistory to take runs from to either "
                               "avoid recalculation or to train the epm")
    req_opts.add_argument("--seed", type=int, help="random seed")
    req_opts.add_argument("--repetitions", default=1, type=int,
                          help="number of repetitions for nondeterministic "
                               "algorithms")
    req_opts.add_argument("--n_jobs", default=1, type=int,
                          help="number of cpu-cores to use")
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

    scenario = Scenario(args_.scenario, cmd_args={'output_dir': ""})
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

    # Load runhistory
    runhistory = RunHistory(average_cost)
    runhistory.load_json(args_.runhistory, scenario.cs)

    if args_.use_epm:
        validator.validate_epm(config_mode=args_.configs,
                               instance_mode=args_.instances,
                               repetitions=args_.repetitions,
                               runhistory=runhistory)
    else:
        validator.validate(config_mode=args_.configs,
                           instance_mode=args_.instances,
                           repetitions=args_.repetitions,
                           n_jobs=args_.n_jobs,
                           runhistory=runhistory,
                           tae=tae)
