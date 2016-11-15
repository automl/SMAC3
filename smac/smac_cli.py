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
from smac.utils.merge_foreign_data import merge_foreign_data

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

        if args_.warmstart_runhistory:
            aggregate_func = average_cost
            rh = RunHistory(aggregate_func=aggregate_func)

            in_scenario = args_.warmstart_scenario
            if not in_scenario:
                in_scenario = scen
            scen, rh = merge_foreign_data(scenario=scen,
                                          runhistory=rh,
                                          in_scenario=in_scenario,
                                          in_runhistory_fn_list=args_.warmstart_runhistory)

        if args_.modus == "SMAC":
            optimizer = SMAC(
                scenario=scen, rng=np.random.RandomState(args_.seed), runhistory=rh)
        elif args_.modus == "ROAR":
            optimizer = ROAR(
                scenario=scen, rng=np.random.RandomState(args_.seed), runhistory=rh)
        optimizer.optimize()

        optimizer.solver.runhistory.save_json(
            fn=os.path.join(scen.output_dir, "runhistory.json"))
        #smbo.runhistory.load_json(fn="runhistory.json", cs=smbo.config_space)
