import os
import sys
import logging
import numpy as np

from smac.utils.io.cmd_reader import CMDReader
from smac.scenario.scenario import Scenario
from smac.smbo.smbo import SMBO

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class SMAC(object):

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

        try:
            smbo = SMBO(scenario=scen, rng=np.random.RandomState(args_.seed))
            smbo.run(max_iters=args_.max_iterations)

        finally:
            smbo.stats.print_stats()
            self.logger.info("Final Incumbent: %s" % (smbo.incumbent))

            smbo.runhistory.save_json()
            #smbo.runhistory.load_json(fn="runhistory.json", cs=smbo.config_space)
