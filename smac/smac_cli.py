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

        scen = Scenario(args_)

        smbo = SMBO(scenario=scen, seed=args_.seed)
        smbo.run(max_iters=2)