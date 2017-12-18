import os
import logging
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS

from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, \
    ChooserLinearCoolDown

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
        req_opts.add_argument("--restore_state", default=None,
                              help="Path to dir with SMAC-files.")
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
        req_opts.add_argument("--random_configuration_chooser", default="", type=str,
                              help="[dev] Constant(<modulus > 1.0>) | "
                                   "Linear(<modulus > 1.0>, <increment >= 0.0>, <end modulus > 1.0>)")
        args_, misc = parser.parse_known_args()
        CMDReader._check_args(args_, set_parsed=True)

        # remove leading '-' in option names
        misc = dict((k.lstrip("-"), v.strip("'"))
                    for k, v in zip(misc[::2], misc[1::2]))

        return args_, misc

    @staticmethod
    def _check_args(args_, set_parsed=False):
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
        CMDReader._check_random_configuration_chooser(args_, set_parsed)

    @staticmethod
    def _check_random_configuration_chooser(args_, set_parsed=False):
        if not hasattr(args_, 'random_configuration_chooser'):
            if set_parsed:
                setattr(args_, 'random_configuration_chooser', None)
            return
        if not args_.random_configuration_chooser:
            if set_parsed:
                args_.random_configuration_chooser = None
            return
        m = re.match(r'^Constant\((.*)\)$', args_.random_configuration_chooser)
        if m:
            try:
                modulus = float(m.group(1))
            except ValueError as e:
                raise ValueError("Constant(modulus): modulus must be a float or int: %s" % (m.group(1)))
            if modulus <= 1.0:
                raise ValueError("Linear: modulus must be > 1.0")
            if set_parsed:
                args_.random_configuration_chooser = ChooserNoCoolDown(modulus)
        else:
            m = re.match(r'^Linear\(([^,]*),([^,]*)(,([^,]*))?\)$', args_.random_configuration_chooser)
            if m:
                groups = list(m.groups())
                try:
                    modulus = float(groups[0])
                except ValueError as e:
                    raise ValueError("Linear(modulus, ...): modulus must be a float or int: %s" % (groups[0]))
                if modulus <= 1.0:
                    raise ValueError("Linear: modulus must be > 1.0")
                try:
                    modulus_increment = float(groups[1])
                except ValueError as e:
                    raise ValueError("Linear(..., increment, ...): increment must be a float or int: %s" % (groups[1]))
                if modulus_increment < 0.0:
                    raise ValueError("Linear: increment must be >= 0.0")
                if not groups[3]:
                    groups[3] = 'inf'
                try:
                    end_modulus = float(groups[3])
                except ValueError as e:
                    raise ValueError("Linear(..., end modulus): end modulus must be a float or int: %s" % (groups[3]))
                if end_modulus <= 1.0:
                    raise ValueError("Linear: end modulus must be > 1.0")
                if set_parsed:
                    args_.random_configuration_chooser = ChooserLinearCoolDown(modulus, modulus_increment, end_modulus)
            else:
                raise ValueError("random_configuration_chooser: invalid format")