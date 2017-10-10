import os
import sys
import logging
import numpy as np
import shutil

from smac.utils.io.cmd_reader import CMDReader
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.facade.roar_facade import ROAR
from smac.facade.epils_facade import EPILS
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.optimizer.objective import average_cost
from smac.utils.merge_foreign_data import merge_foreign_data_from_file
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.io.input_reader import InputReader
from smac.tae.execute_ta_run import TAEAbortException, FirstRunCrashedException

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class SMACCLI(object):

    """Main class of SMAC"""

    def __init__(self):
        """Constructor"""
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

    def main_cli(self):
        """Main function of SMAC for CLI interface"""
        self.logger.info("SMAC call: %s" % (" ".join(sys.argv)))

        cmd_reader = CMDReader()
        args_, misc_args = cmd_reader.read_cmd()

        root_logger = logging.getLogger()
        root_logger.setLevel(args_.verbose_level)
        logger_handler = logging.StreamHandler(
                stream=sys.stdout)
        if root_logger.level >= logging.INFO:
            formatter = logging.Formatter(
                "%(levelname)s:\t%(message)s")
        else:
            formatter = logging.Formatter(
                "%(asctime)s:%(levelname)s:%(name)s:%(message)s",
                "%Y-%m-%d %H:%M:%S")
        logger_handler.setFormatter(formatter)
        root_logger.addHandler(logger_handler)
        # remove default handler
        root_logger.removeHandler(root_logger.handlers[0])

        scen = Scenario(args_.scenario_file, misc_args,
                        run_id=args_.seed)

        rh = None
        if args_.warmstart_runhistory:
            aggregate_func = average_cost
            rh = RunHistory(aggregate_func=aggregate_func)

            scen, rh = merge_foreign_data_from_file(
                scenario=scen,
                runhistory=rh,
                in_scenario_fn_list=args_.warmstart_scenario,
                in_runhistory_fn_list=args_.warmstart_runhistory,
                cs=scen.cs,
                aggregate_func=aggregate_func)

        initial_configs = None
        if args_.warmstart_incumbent:
            initial_configs = [scen.cs.get_default_configuration()]
            for traj_fn in args_.warmstart_incumbent:
                trajectory = TrajLogger.read_traj_aclib_format(
                    fn=traj_fn, cs=scen.cs)
                initial_configs.append(trajectory[-1]["incumbent"])

        # Restore state
        stats = None
        incumbent = None
        if args_.restore_state:
            root_logger.debug("Restoring state from %s...", args_.restore_state)
            rh, stats, incumbent = self.restore_state(args_, scen, root_logger)

        if args_.mode == "SMAC":
            optimizer = SMAC(
                scenario=scen,
                rng=np.random.RandomState(args_.seed),
                runhistory=rh,
                initial_configurations=initial_configs,
                stats=stats,
                restore_incumbent=incumbent)
        elif args_.mode == "ROAR":
            optimizer = ROAR(
                scenario=scen,
                rng=np.random.RandomState(args_.seed),
                runhistory=rh,
                initial_configurations=initial_configs)
        elif args_.mode == "EPILS":
            optimizer = EPILS(
                scenario=scen,
                rng=np.random.RandomState(args_.seed),
                runhistory=rh,
                initial_configurations=initial_configs)
        try:
            optimizer.optimize()
        except (TAEAbortException, FirstRunCrashedException) as err:
            self.logger.error(err)

    def restore_state(self, args_, scen, root_logger):
        # Check for folder and files
        rh_path = os.path.join(args_.restore_state, "runhistory.json")
        stats_path = os.path.join(args_.restore_state, "stats.json")
        traj_path = os.path.join(args_.restore_state, "traj_aclib2.json")
        scen_path = os.path.join(args_.restore_state, "scenario.txt")
        if not os.path.isdir(args_.restore_state):
           raise FileNotFoundError("Could not find folder from which to restore.")
        # Load runhistory and stats
        rh = RunHistory(aggregate_func=None)
        rh.load_json(rh_path, scen.cs)
        root_logger.debug("Restored runhistory from %s", rh_path)
        stats = Stats(scen)
        stats.load(stats_path)
        root_logger.debug("Restored stats from %s", stats_path)
        trajectory = TrajLogger.read_traj_aclib_format(
            fn=traj_path, cs=scen.cs)
        incumbent = trajectory[-1]["incumbent"]
        root_logger.debug("Restored incumbent %s from %s", incumbent, traj_path)
        # Copy traj if output_dir of specified scenario-file is different than
        # the output_dir of the scenario-file in the folder from which to restore.
        if scen.output_dir != InputReader().read_scenario_file(scen_path)['output_dir']:
            new_traj_path = os.path.join(scen.output_dir, "traj_aclib2.json")
            shutil.copy(traj_path, new_traj_path)
            root_logger.debug("Copied traj %s", rh_path)
        return rh, stats, incumbent
