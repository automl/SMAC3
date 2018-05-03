import os
import logging
import typing
from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser, FileType, Namespace, SUPPRESS

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


help_type = "standard"


class ConfigurableHelpFormatter(ArgumentDefaultsHelpFormatter):
    """
    Configurable Help Formatter. Can filter out developer options.
    """

    def __init__(self, *args, **kwargs):
        super(ConfigurableHelpFormatter, self).__init__(*args, **kwargs)

    def _add_item(self, func: typing.Callable, args: typing.Union[list, tuple]):
        def filter_actions(actions: typing.List[Action]):
            filtered_actions = []
            for action in actions:
                dev = False
                if isinstance(action.help, str):
                    if action.help.startswith('[dev]'):
                        dev = True
                else:
                    for s in action.option_strings:
                        if s.startswith('--dev'):
                            dev = True
                            break
                if not dev:
                    filtered_actions.append(action)
            return filtered_actions
        if help_type == 'standard':
            if func.__name__ == '_format_usage':
                args = (args[0], filter_actions(args[1]), args[2], args[3])
            elif isinstance(args, list):
                if len(args):
                    args = filter_actions(args)
                    if not len(args):
                        return
        self._current_section.items.append((func, args))


class StandardHelpAction(Action):
    """Action to only show standard options in help message"""

    def __init__(self, *args, **kwargs):
        super(StandardHelpAction, self).__init__(default=SUPPRESS, nargs=0, *args, **kwargs)

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        global help_type
        help_type = 'standard'
        parser.print_help()
        parser.exit()


class DevHelpAction(Action):
    """Action to show standard and developer options in help message"""

    def __init__(self, *args, **kwargs):
        super(DevHelpAction, self).__init__(default=SUPPRESS, nargs=0, *args, **kwargs)

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        global help_type
        help_type = 'dev'
        parser.print_help()
        parser.exit()


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

        parser = ArgumentParser(formatter_class=ConfigurableHelpFormatter, add_help=False)
        req_opts = parser.add_argument_group("Required Options")
        req_opts.add_argument("--scenario_file", required=True,
                              help="scenario file in AClib format")

        # let a help message begin with "[dev]" to add a developer option
        req_opts = parser.add_argument_group("Optional Options")
        req_opts.add_argument("--help", action=StandardHelpAction,
                              help="Show help messages for standard options.")
        req_opts.add_argument("--help-all", action=DevHelpAction,
                              help="Show help messages for both standard and developer options.")
        req_opts.add_argument("--seed", default=1, type=int,
                              help="random seed")
        req_opts.add_argument("--verbose_level", default=logging.INFO,
                              choices=["INFO", "DEBUG"],
                              help="verbose level")
        req_opts.add_argument("--mode", default="SMAC",
                              choices=["SMAC", "ROAR", "EPILS", "Hydra"],
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
        # list of trajectory dump files, reads runhistory and uses final incumbent as challenger
        req_opts.add_argument("--warmstart_incumbent", default=None,
                              nargs="*",
                              help=SUPPRESS)
        req_opts.add_argument("--random_configuration_chooser", default=None, type=FileType('r'),
                              help="[dev] path to a python module containing a class `RandomConfigurationChooserImpl`"
                                   "implementing the interface of `RandomConfigurationChooser`")
        req_opts.add_argument("--hydra_iterations",
                              default=3,
                              type=int,
                              help="[dev] number of hydra iterations. Only active if mode is set to Hydra")
        req_opts.add_argument("--hydra_validation",
                              default='train',
                              choices=['train', 'val10', 'val20'],
                              type=str.lower,
                              help="[dev] set to validate incumbents on. valX =>"
                                   " validation set of size training_set * 0.X")
        req_opts.add_argument("--hydra_incumbents_per_round",
                              default=1,
                              type=int,
                              help="[dev] number of configurations to keep per hydra iteration.")
        req_opts.add_argument("--hydra_n_optimizers",
                              default=1,
                              type=int,
                              help="[dev] number of optimizers to run in parallel per hydra iteration.")

        args_, misc = parser.parse_known_args()
        CMDReader._check_args(args_, set_parsed=True)

        # remove leading '-' in option names
        misc = dict((k.lstrip("-"), v.strip("'"))
                    for k, v in zip(misc[::2], misc[1::2]))

        return args_, misc

    @staticmethod
    def _check_args(args_: Namespace, set_parsed: bool=False):
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
    def _check_random_configuration_chooser(args_: Namespace, set_parsed: bool=False):
        if not hasattr(args_, 'random_configuration_chooser'):
            if set_parsed:
                setattr(args_, 'random_configuration_chooser', None)
            return
        elif not args_.random_configuration_chooser:
            if set_parsed:
                args_.random_configuration_chooser = None
            return
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location("smac.custom.random_configuration_chooser", args_.random_configuration_chooser.name)
            rcc_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rcc_module)
            if set_parsed:
                setattr(args_, 'random_configuration_chooser', rcc_module.RandomConfigurationChooserImpl())
