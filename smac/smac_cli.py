import typing

import logging
import os
import sys

import numpy as np

from smac.configspace import Configuration
from smac.facade.experimental.hydra_facade import (  # type: ignore[attr-defined] # noqa F821
    Hydra,
)
from smac.facade.experimental.psmac_facade import (  # type: ignore[attr-defined] # noqa F821
    PSMAC,
)
from smac.facade.roar_facade import ROAR
from smac.facade.smac_ac_facade import SMAC4AC
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae import FirstRunCrashedException, TAEAbortException
from smac.utils.io.cmd_reader import CMDReader
from smac.utils.io.output_directory import create_output_directory
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.merge_foreign_data import merge_foreign_data_from_file

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class SMACCLI(object):
    """Main class of SMAC."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

    def main_cli(self, commandline_arguments: typing.Optional[typing.List[str]] = None) -> None:
        """Main function of SMAC for CLI interface."""
        self.logger.info("SMAC call: %s" % (" ".join(sys.argv)))

        cmd_reader = CMDReader()
        kwargs = {}
        if commandline_arguments:
            kwargs["commandline_arguments"] = commandline_arguments
        main_args_, smac_args_, scen_args_ = cmd_reader.read_cmd(**kwargs)

        root_logger = logging.getLogger()
        root_logger.setLevel(main_args_.verbose_level)
        logger_handler = logging.StreamHandler(stream=sys.stdout)
        if root_logger.level >= logging.INFO:
            formatter = logging.Formatter("%(levelname)s:\t%(message)s")
        else:
            formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:\t%(message)s", "%Y-%m-%d %H:%M:%S")
        logger_handler.setFormatter(formatter)
        root_logger.addHandler(logger_handler)
        # remove default handler
        if len(root_logger.handlers) > 1:
            root_logger.removeHandler(root_logger.handlers[0])

        # Create defaults
        rh = None
        initial_configs = None
        stats = None
        incumbent = None

        # Create scenario-object
        scenario = {}
        scenario.update(vars(smac_args_))
        scenario.update(vars(scen_args_))
        scen = Scenario(scenario=scenario)

        # Restore state
        if main_args_.restore_state:
            root_logger.debug("Restoring state from %s...", main_args_.restore_state)
            restore_state = main_args_.restore_state
            rh, stats, traj_list_aclib, traj_list_old = self.restore_state(scen, restore_state)

            scen.output_dir_for_this_run = create_output_directory(
                scen,
                main_args_.seed,
                root_logger,
            )
            scen.write()
            incumbent = self.restore_state_after_output_dir(scen, stats, traj_list_aclib, traj_list_old)

        if main_args_.warmstart_runhistory:
            rh = RunHistory()

            scen, rh = merge_foreign_data_from_file(
                scenario=scen,
                runhistory=rh,
                in_scenario_fn_list=main_args_.warmstart_scenario,
                in_runhistory_fn_list=main_args_.warmstart_runhistory,
                cs=scen.cs,  # type: ignore[attr-defined] # noqa F821
            )

        if main_args_.warmstart_incumbent:
            initial_configs = [scen.cs.get_default_configuration()]  # type: ignore[attr-defined] # noqa F821
            for traj_fn in main_args_.warmstart_incumbent:
                trajectory = TrajLogger.read_traj_aclib_format(
                    fn=traj_fn,
                    cs=scen.cs,  # type: ignore[attr-defined] # noqa F821
                )
                initial_configs.append(trajectory[-1]["incumbent"])

        if main_args_.mode == "SMAC4AC":
            optimizer = SMAC4AC(
                scenario=scen,
                rng=np.random.RandomState(main_args_.seed),
                runhistory=rh,
                initial_configurations=initial_configs,
                stats=stats,
                restore_incumbent=incumbent,
                run_id=main_args_.seed,
            )
        elif main_args_.mode == "SMAC4HPO":
            optimizer = SMAC4HPO(
                scenario=scen,
                rng=np.random.RandomState(main_args_.seed),
                runhistory=rh,
                initial_configurations=initial_configs,
                stats=stats,
                restore_incumbent=incumbent,
                run_id=main_args_.seed,
            )
        elif main_args_.mode == "SMAC4BB":
            optimizer = SMAC4BB(
                scenario=scen,
                rng=np.random.RandomState(main_args_.seed),
                runhistory=rh,
                initial_configurations=initial_configs,
                stats=stats,
                restore_incumbent=incumbent,
                run_id=main_args_.seed,
            )
        elif main_args_.mode == "ROAR":
            optimizer = ROAR(
                scenario=scen,
                rng=np.random.RandomState(main_args_.seed),
                runhistory=rh,
                initial_configurations=initial_configs,
                run_id=main_args_.seed,
            )
        elif main_args_.mode == "Hydra":
            optimizer = Hydra(
                scenario=scen,
                rng=np.random.RandomState(main_args_.seed),
                runhistory=rh,
                initial_configurations=initial_configs,
                stats=stats,
                restore_incumbent=incumbent,
                run_id=main_args_.seed,
                random_configuration_chooser=main_args_.random_configuration_chooser,
                n_iterations=main_args_.hydra_iterations,
                val_set=main_args_.hydra_validation,
                incs_per_round=main_args_.hydra_incumbents_per_round,
                n_optimizers=main_args_.hydra_n_optimizers,
            )
        elif main_args_.mode == "PSMAC":
            optimizer = PSMAC(
                scenario=scen,
                rng=np.random.RandomState(main_args_.seed),
                run_id=main_args_.seed,
                shared_model=smac_args_.shared_model,
                validate=main_args_.psmac_validate,
                n_optimizers=main_args_.hydra_n_optimizers,
                n_incs=main_args_.hydra_incumbents_per_round,
            )
        try:
            optimizer.optimize()
        except (TAEAbortException, FirstRunCrashedException) as err:
            self.logger.error(err)

    def restore_state(
        self,
        scen: Scenario,
        restore_state: str,
    ) -> typing.Tuple[RunHistory, Stats, typing.List, typing.List]:
        """Read in files for state-restoration: runhistory, stats, trajectory."""
        # Check for folder and files
        rh_path = os.path.join(restore_state, "runhistory.json")
        stats_path = os.path.join(restore_state, "stats.json")
        traj_path_aclib = os.path.join(restore_state, "traj_aclib2.json")
        traj_path_old = os.path.join(restore_state, "traj_old.csv")
        _ = os.path.join(restore_state, "scenario.txt")
        if not os.path.isdir(restore_state):
            raise FileNotFoundError("Could not find folder from which to restore.")
        # Load runhistory and stats
        rh = RunHistory()
        rh.load_json(rh_path, scen.cs)  # type: ignore[attr-defined] # noqa F821
        self.logger.debug("Restored runhistory from %s", rh_path)
        stats = Stats(scen)
        stats.load(stats_path)
        self.logger.debug("Restored stats from %s", stats_path)
        with open(traj_path_aclib, "r") as traj_fn:
            traj_list_aclib = traj_fn.readlines()
        with open(traj_path_old, "r") as traj_fn:
            traj_list_old = traj_fn.readlines()
        return rh, stats, traj_list_aclib, traj_list_old

    def restore_state_after_output_dir(
        self,
        scen: Scenario,
        stats: Stats,
        traj_list_aclib: typing.List,
        traj_list_old: typing.List,
    ) -> Configuration:
        """Finish processing files for state-restoration.

        Trajectory is read in, but needs to be written to new output-
        folder. Therefore, the output-dir is created. This needs to be
        considered in the SMAC-facade.
        """
        # write trajectory-list
        traj_path_aclib = os.path.join(scen.output_dir, "traj_aclib2.json")  # type: ignore[attr-defined] # noqa F821
        traj_path_old = os.path.join(scen.output_dir, "traj_old.csv")  # type: ignore[attr-defined] # noqa F821
        with open(traj_path_aclib, "w") as traj_fn:
            traj_fn.writelines(traj_list_aclib)
        with open(traj_path_old, "w") as traj_fn:
            traj_fn.writelines(traj_list_old)
        # read trajectory to retrieve incumbent
        # TODO replace this with simple traj_path_aclib?
        trajectory = TrajLogger.read_traj_aclib_format(fn=traj_path_aclib, cs=scen.cs)  # type: ignore[attr-defined] # noqa F821
        incumbent = trajectory[-1]["incumbent"]
        self.logger.debug("Restored incumbent %s from %s", incumbent, traj_path_aclib)
        return incumbent


def cmd_line_call() -> None:
    """Entry point to be installable to /user/bin."""
    smac = SMACCLI()
    smac.main_cli()
