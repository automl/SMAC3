import os
import sys
import logging
import numpy as np

from utils.io.cmd_reader import CMDReader
from scenario.scenario import Scenario
from smbo.smbo import SMBO
from stats.stats import Stats

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "GPLv3"
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
        args_ = cmd_reader.read_cmd()

        logging.basicConfig(level=args_.verbose_level)

        # TODO: hack to set logger level as long as some of our dependencies does not handle logger
        # in a correct ways
        if args_.verbose_level == "DEBUG":
            self.logger.parent.level = 10

        scen = Scenario(args_.scenario_file)

        # necessary to use stats options related to scenario information
        Stats.scenario = scen

        try:
            smbo = SMBO(scenario=scen, rng=np.random.RandomState(args_.seed))
            smbo.run(max_iters=args_.max_iterations)

        finally:
            Stats.print_stats()
            self.logger.info("Final Incumbent: %s" % (smbo.incumbent))
 
            smbo.runhistory.save_json()
            #smbo.runhistory.load_json(fn="runhistory.json", cs=smbo.config_space)
