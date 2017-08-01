import os
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class CMDReader(object):

    """Use argparse to parse command line options

    Attributes
    ----------
    logger : Logger
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        pass

    def read_cmd(self):
        """Reads command line options

        Returns
        -------
            args_: parsed arguments; return of parse_args of ArgumentParser
        """

        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        req_opts = parser.add_argument_group("Required Options")
        req_opts.add_argument("--scenario_file", required=True,
                              help="scenario file in AClib format")

        req_opts = parser.add_argument_group("Optional Options")
        req_opts.add_argument("--seed", default=1, type=int,
                              help="random seed")
        req_opts.add_argument("--verbose_level", default=logging.INFO,
                              choices=["INFO", "DEBUG"],
                              help="verbose level")
        req_opts.add_argument("--mode", default="SMAC",
                              choices=["SMAC", "ROAR", "EPILS"],
                              help="Configuration mode.")
        req_opts.add_argument("--warmstart_runhistory", default=None,
                              nargs="*",
                              help=SUPPRESS)  # list of runhistory dump files
        # scenario corresponding to --warmstart_runhistory; 
        # pcs and feature space has to be identical to --scenario_file
        req_opts.add_argument("--warmstart_scenario", default=None,
                              nargs="*",
                              help=SUPPRESS)  
        req_opts.add_argument("--warmstart_incumbent", default=None,
                              nargs="*",
                              help=SUPPRESS)# list of trajectory dump files, 
                                            # reads runhistory 
                                            # and uses final incumbent as challenger 

        args_, misc = parser.parse_known_args()
        self._check_args(args_)

        # remove leading '-' in option names
        misc = dict((k.lstrip("-"), v.strip("'"))
                    for k, v in zip(misc[::2], misc[1::2]))

        return args_, misc

    def _check_args(self, args_):
        """Checks command line arguments (e.g., whether all given files exist)

        Parameters
        ----------
        args_: parsed arguments
            Parsed command line arguments

        Raises
        ------
        ValueError
            in case of missing files or wrong configurations
        """

        if not os.path.isfile(args_.scenario_file):
            raise ValueError("Not found: %s" % (args_.scenario_file))
