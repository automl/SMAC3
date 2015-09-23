'''
Created on Sep 23, 2015

@author: lindauer
'''

import os
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class CMDReader(object):
    '''
        use argparse to parse command line options
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.logger = logging.getLogger("CMDReader")
        pass

    def read_cmd(self):
        '''
            reads command line options
            Returns:
                parsed arguements
        '''
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        req_opts = parser.add_argument_group("Required Options")
        req_opts.add_argument("--scenario_file", required=True,
                              help="scenario file in AClib format")

        req_opts = parser.add_argument_group("Optional Options")
        req_opts.add_argument("--seed", default=12345, type=int,
                              help="random seed")

        args_ = parser.parse_args()
        self.check_args(args_)

        return args_

    def _check_args(self, args_):
        '''
            checks command line arguments
            (e.g., whether all given files exist)
            Raises Errors in case of problems
        '''
        if not os.path.isfile(args_.scenario_file):
            raise ValueError("Not found: %s" % (args_.scenario_file))
