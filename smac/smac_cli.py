import os
import sys
import logging

from utils.io.cmd_reader import CMDReader
from scenario.scenario import Scenario
from smbo.smbo import SMBO

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
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

        # TODO: hack to set logger level as long as robo does not handle logger
        # in a correct ways
        if args_.verbose_level == "DEBUG":
            self.logger.parent.level = 10

        scen = Scenario(args_)

        smbo = SMBO(scenario=scen, seed=args_.seed)
        smbo.run(max_iters=args_.max_iterations)
        
        self.logger.info("Final Incumbent: %s" %(smbo.incumbent))
