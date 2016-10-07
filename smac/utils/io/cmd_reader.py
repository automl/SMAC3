__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"

import os
import logging
import numpy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS


class CMDReader(object):

    """
        use argparse to parse command line options

        Attributes
        ----------
        logger : Logger oject
    """

    def __init__(self):
        """
        Constructor
        """
        self.logger = logging.getLogger("CMDReader")
        pass

    def read_cmd(self):
        """
            reads command line options

            Returns
            -------
                args_: parsed arguments; return of parse_args of ArgumentParser
        """

        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        req_opts = parser.add_argument_group("Required Options")
        req_opts.add_argument("--scenario_file", required=True,
                              help="scenario file in AClib format")

        req_opts = parser.add_argument_group("Optional Options")
        req_opts.add_argument("--seed", default=12345, type=int,
                              help="random seed")
        req_opts.add_argument("--verbose_level", default=logging.INFO,
                              choices=["INFO", "DEBUG"],
                              help="random seed")
        req_opts.add_argument("--modus", default="SMAC",
                              choices=["SMAC", "ROAR"],
                              help=SUPPRESS)

        args_, misc = parser.parse_known_args()
        self._check_args(args_)

        # remove leading '-' in option names
        misc = dict((k.lstrip("-"), v.strip("'"))
                    for k, v in zip(misc[::2], misc[1::2]))

        return args_, misc

    def _check_args(self, args_):
        """
            checks command line arguments
            (e.g., whether all given files exist)

            Parameters
            ----------
            args_: parsed arguments
                parsed command line arguments

            Raises
            ------
            ValueError
                in case of missing files or wrong configurations
        """

        if not os.path.isfile(args_.scenario_file):
            raise ValueError("Not found: %s" % (args_.scenario_file))
