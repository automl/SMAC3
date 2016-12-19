import os
import sys
import logging
import numpy as np

from smac.utils.io.cmd_reader import CMDReader
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.facade.roar_facade import ROAR
from smac.runhistory.runhistory import RunHistory
from smac.smbo.objective import average_cost
from smac.utils.merge_foreign_data import merge_foreign_data_from_file
from smac.utils.io.traj_logging import TrajLogger

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class SMACCLI(object):

    '''
    main class of SMAC
    '''

    def __init__(self):
        '''
            constructor
        '''
        self.logger = logging.getLogger("SMAC")

    def main_cli(self):
        '''
            main function of SMAC for CLI interface
        '''

        cmd_reader = CMDReader()
        args_, misc_args = cmd_reader.read_cmd()

        logging.basicConfig(level=args_.verbose_level)

        root_logger = logging.getLogger()
        root_logger.setLevel(args_.verbose_level)

        scen = Scenario(args_.scenario_file, misc_args)

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
                trajectory = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=scen.cs)
                initial_configs.append(trajectory[-1]["incumbent"])
                
        if args_.modus == "SMAC":
            optimizer = SMAC(
                scenario=scen, 
                rng=np.random.RandomState(args_.seed), 
                runhistory=rh, 
                initial_configurations=initial_configs)
        elif args_.modus == "ROAR":
            optimizer = ROAR(
                scenario=scen, 
                rng=np.random.RandomState(args_.seed), 
                runhistory=rh, 
                initial_configurations=initial_configs)
        try:
            optimizer.optimize()

        finally:
            # ensure that the runhistory is always dumped in the end
            if scen.output_dir is not None:
                optimizer.solver.runhistory.save_json(
                    fn=os.path.join(scen.output_dir, "runhistory.json"))
        #smbo.runhistory.load_json(fn="runhistory.json", cs=smbo.config_space)
