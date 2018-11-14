from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser, FileType, Namespace, SUPPRESS
import datetime
import distutils.util
import logging
import os
import re
import shlex
import sys
from smac.configspace import pcs, pcs_new
from smac.utils.constants import MAXINT, N_TREES
from smac.utils.io.input_reader import InputReader
import time
import typing

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"

in_reader = InputReader()
parsed_scen_args = {}
logger = None

def truthy(x):
    """Convert x into its truth value"""
    if isinstance(x, bool):
        return x
    elif isinstance(x, (int, float)):
        return x != 0
    elif isinstance(x, str):
        return bool(distutils.util.strtobool(x))
    else:
        return False


class CheckScenarioFileAction(Action):
    """Check scenario file given by user"""

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        fn = values
        if fn:
            if not os.path.isfile(fn):
                parser.exit(1, "Could not find scenario file: {}".format(fn))
        setattr(namespace, self.dest, values)


class ParseRandomConfigurationChooserAction(Action):
    """Parse random configuration chooser given by user"""

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: list, option_string: str = None):
        module_file = values
        module_path = module_file.name
        module_file.close()
        import importlib.util
        spec = importlib.util.spec_from_file_location("smac.custom.random_configuration_chooser", module_path)
        rcc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rcc_module)
        setattr(namespace, self.dest, rcc_module.RandomConfigurationChooserImpl())


class ProcessRunObjectiveAction(Action):
    """Process run objective given by user"""

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        if values == "runtime":
            parsed_scen_args["cutoff_time_required"] = {
                "error": "--cutoff-time is required when --run-objective is set to \"runtime\""
            }
        setattr(namespace, self.dest, values)


class ParseOverallObjectiveAction(Action):
    """Parse overall objective given by user"""

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        par_str = values
        if par_str[:3] in ["PAR", "par"]:
            par_str = par_str[3:]
        elif par_str[:4] in ["mean", "MEAN"]:
            par_str = par_str[4:]
        # Check for par-value as in "par10"/ "mean5"
        if par_str:
            parsed_scen_args["par_factor"] = int(par_str)
        else:
            logger.debug("No par-factor detected. Using 1 by default.")
            parsed_scen_args["par_factor"] = 1
        setattr(namespace, self.dest, values)


class ReadTrainInstFileAction(Action):
    """Read training instance file given by user"""

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        fn = values
        if fn:
            if os.path.isfile(fn):
                parsed_scen_args["train_insts"] = in_reader.read_instance_file(fn)
            else:
                parser.exit(1, "Could not find instance file: {}".format(fn))
        setattr(namespace, self.dest, values)


class ReadTestInstFileAction(Action):
    """Read test instance file given by user"""

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        fn = values
        if fn:
            if os.path.isfile(fn):
                parsed_scen_args["test_insts"] = in_reader.read_instance_file(fn)
            else:
                parser.exit(1, "Could not find test instance file: {}".format(fn))
        setattr(namespace, self.dest, values)


class ReadFeatureFileAction(Action):
    """Read feature file given by user"""

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        fn = values
        if fn:
            if os.path.isfile(fn):
                parsed_scen_args["features"] = in_reader.read_instance_features_file(fn)
                parsed_scen_args["feature_names"], parsed_scen_args["feature_dict"] = parsed_scen_args["features"]
            else:
                parser.exit(1, "Could not find feature file: {}".format(fn))
        setattr(namespace, self.dest, values)


class ReadPCSFileAction(Action):
    """Read PCS (parameter configuration space) file given by user"""

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        fn = values
        if fn:
            if os.path.isfile(fn):
                with open(fn) as fp:
                    pcs_str = fp.readlines()
                    try:
                        parsed_scen_args["cs"] = pcs.read(pcs_str)
                    except:
                        logger.debug("Could not parse pcs file with old format; trying new format ...")
                        parsed_scen_args["cs"] = pcs_new.read(pcs_str)
                    parsed_scen_args["cs"].seed(42)
            else:
                parser.exit(1, "Could not find pcs file: {}".format(fn))
        setattr(namespace, self.dest, values)


class ProcessOutputDirAction(Action):
    """Process output directory given by user"""

    def __call__(self, parser: ArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        directory = values
        if not directory:
            logger.debug("Deactivate output directory.")
            values = None
        else:
            logger.info("Output to {}".format(directory))
        setattr(namespace, self.dest, values)


class ConfigurableHelpFormatter(ArgumentDefaultsHelpFormatter):
    """
    Configurable Help Formatter. Can filter out developer options.
    """

    def __init__(self, *args, help_type='standard', **kwargs):
        self.help_type = help_type
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

        if self.help_type == 'standard':
            if func.__name__ == '_format_usage':
                args = (args[0], filter_actions(args[1]), args[2], args[3])
            elif isinstance(args, list):
                if args:
                    args = filter_actions(args)
                    if not args:
                        return
        self._current_section.items.append((func, args))


class SMACArgumentParser(ArgumentParser):
    """
    ArgumentParser that can be extended by additional parsers.
    """

    def __init__(self, *args, **kwargs):
        self.additional_parsers = []
        self.help_type = 'standard'  # standard or dev
        super(SMACArgumentParser, self).__init__(*args, **kwargs)

    def set_help_type(self, help_type: str):
        self.help_type = help_type
        for parser in self.additional_parsers:
            parser.help_type = help_type

    def add_parser(self, additional_parser: ArgumentParser):
        self.additional_parsers.append(additional_parser)

    def _get_formatter(self):
        return self.formatter_class(prog=self.prog, help_type=self.help_type)

    def format_help(self):
        formatter = self._get_formatter()

        # usage
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)

        # description
        formatter.add_text(self.description)

        # positionals, optionals and user-defined groups
        def add_action_groups(parser: ArgumentParser):
            for action_group in parser._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
        add_action_groups(self)

        # positionals, optionals and user-defined groups from additional parsers
        for parser in self.additional_parsers:
            add_action_groups(parser)

        # epilog
        formatter.add_text(self.epilog)

        # determine help from format above
        return formatter.format_help()


class StandardHelpAction(Action):
    """Action to only show standard options in help message"""

    def __init__(self, *args, **kwargs):
        super(StandardHelpAction, self).__init__(default=SUPPRESS, nargs=0, *args, **kwargs)

    def __call__(self, parser: SMACArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        parser.set_help_type('standard')
        parser.print_help()
        parser.exit()


class DevHelpAction(Action):
    """Action to show standard and developer options in help message"""

    def __init__(self, *args, **kwargs):
        super(DevHelpAction, self).__init__(default=SUPPRESS, nargs=0, *args, **kwargs)

    def __call__(self, parser: SMACArgumentParser, namespace: Namespace, values: list, option_string: str=None):
        parser.set_help_type('dev')
        parser.print_help()
        parser.exit()


class CMDReader(object):
    """Use argparse to parse command line options

    Attributes
    ----------
    logger : Logger
    """

    def __init__(self):
        global logger, parsed_scen_args
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        logger = self.logger

        # initialized in _add_main_options
        self.parser = None
        self.main_cmd_actions = {}
        self.main_cmd_translations = {}
        # initialized in _add_smac_options
        self.smac_parser = None
        self.smac_cmd_actions = {}
        self.smac_cmd_translations = {}
        # initialized in _add_scen_options
        self.scen_parser = None
        self.scen_cmd_actions = {}
        self.scen_cmd_translations = {}
        # needed for argument interdependencies
        self.parsed_scen_args = {}
        parsed_scen_args = self.parsed_scen_args

        # add arguments to parser
        self._add_main_options()
        self._add_smac_options()
        self._add_scen_options()

    @staticmethod
    def _extract_action_info(actions: typing.List[Action]):
        extracted_info = {}
        translations = {}
        for action in actions:
            name = list(filter(lambda e: e.startswith('--'), action.option_strings))
            if name:
                name = name[0]
            else:
                name = action.option_strings[0]
            dest = name
            if hasattr(action, 'dest'):
                dest = action.dest
            cmd_action = {
                'dest': dest
            }
            for name in action.option_strings:
                translations[name] = dest
                translations[name.lstrip('-')] = dest
            if hasattr(action, 'type'):
                cmd_action['type'] = action.type
            else:
                cmd_action['type'] = str
            if hasattr(action, 'default'):
                if action.default == SUPPRESS:
                    continue
                cmd_action['default'] = action.default
            else:
                cmd_action['default'] = None
            if hasattr(action, 'choices'):
                cmd_action['choices'] = action.choices
            else:
                cmd_action['choices'] = None
            if hasattr(action, 'required'):
                cmd_action['required'] = action.required
            else:
                cmd_action['required'] = False
            if hasattr(action, 'help'):
                cmd_action['help'] = action.help
            else:
                cmd_action['help'] = None
            cmd_action['option_strings'] = action.option_strings
            extracted_info[name] = cmd_action
        return extracted_info, translations

    def _add_main_options(self):
        """Add main Options"""
        prog = sys.argv[0]
        if re.match("^python[0-9._-]*$", sys.argv[0]):
            prog = sys.argv[1]
        self.parser = SMACArgumentParser(formatter_class=ConfigurableHelpFormatter, add_help=False, prog=prog)
        # let a help message begin with "[dev]" to add a developer option
        req_opts = self.parser.add_argument_group("Required Options")
        req_opts.add_argument("--scenario", "--scenario-file", "--scenario_file", dest="scenario_file",
                              required=True, type=str,
                              action=CheckScenarioFileAction,
                              help="Scenario file in AClib format.")
        opt_opts = self.parser.add_argument_group("Optional Options")
        opt_opts.add_argument("--help", action=StandardHelpAction,
                              help="Show help messages for standard options.")
        opt_opts.add_argument("--help-all", action=DevHelpAction,
                              help="Show help messages for both standard and developer options.")
        opt_opts.add_argument("--seed",
                              default=1, type=int,
                              help="Random Seed.")
        opt_opts.add_argument("--verbose", "--verbose-level", "--verbose_level", dest="verbose_level",
                              default=logging.INFO, choices=["INFO", "DEBUG"],
                              help="Verbosity level.")
        opt_opts.add_argument("--mode",
                              default="SMAC", choices=["SMAC", "ROAR", "EPILS", "Hydra", "PSMAC", "BORF", "BOGP"],
                              help="Configuration mode.")
        opt_opts.add_argument("--restore-state", "--restore_state", dest="restore_state",
                              default=None,
                              help="Path to directory with SMAC-files.")
        # list of runhistory dump files
        # scenario corresponding to --warmstart_runhistory;
        # pcs and feature space has to be identical to --scenario_file
        opt_opts.add_argument("--warmstart-runhistory", "--warmstart_runhistory", dest="warmstart_runhistory",
                              default=None, nargs="*",
                              help=SUPPRESS)
        opt_opts.add_argument("--warmstart-scenario", "--warmstart_scenario", dest="warmstart_scenario",
                              default=None, nargs="*",
                              help=SUPPRESS)
        # list of trajectory dump files, reads runhistory and uses final incumbent as challenger
        opt_opts.add_argument("--warmstart-incumbent", "--warmstart_incumbent", dest="warmstart_incumbent",
                              default=None, nargs="*",
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
                              choices=['train', 'val10', 'val20', 'val30', 'val40', 'val50', 'none'],
                              type=str.lower,
                              help="[dev] set to validate incumbents on. valX =>"
                                   " validation set of size training_set * 0.X")
        req_opts.add_argument("--incumbents_per_round",
                              default=1,
                              type=int,
                              help="[dev] number of configurations to keep per psmac/hydra iteration.",
                              dest="hydra_incumbents_per_round")
        req_opts.add_argument("--n_optimizers",
                              default=1,
                              type=int,
                              help="[dev] number of optimizers to run in parallel per psmac/hydra iteration.",
                              dest="hydra_n_optimizers")
        req_opts.add_argument("--psmac_validate",
                              default=False, type=truthy,
                              help="[dev] Validate all psmac configurations.")

        self.main_cmd_actions, self.main_cmd_translations = CMDReader._extract_action_info(self.parser._actions)

    def _add_smac_options(self):
        """Add SMAC Options"""
        self.smac_parser = SMACArgumentParser(formatter_class=ConfigurableHelpFormatter, add_help=False)
        smac_opts = self.smac_parser.add_argument_group("SMAC Options")
        smac_opts.add_argument("--abort-on-first-run-crash", "--abort_on_first_run_crash",
                               dest='abort_on_first_run_crash',
                               default=True, type=truthy,
                               help="If true, *SMAC* will abort if the first run of "
                                    "the target algorithm crashes.")

        smac_opts.add_argument("--minr", "--minR", dest='minR',
                               default=1, type=int,
                               help="[dev] Minimum number of calls per configuration.")
        smac_opts.add_argument("--maxr", "--maxR", dest='maxR',
                               default=2000, type=int,
                               help="[dev] Maximum number of calls per configuration.")
        self.output_dir_arg = \
            smac_opts.add_argument("--output-dir", "--output_dir", dest='output_dir',
                                   type=str, action=ProcessOutputDirAction,
                                   default="smac3-output_%s" % (
                                       datetime.datetime.fromtimestamp(
                                           time.time()).strftime(
                                           '%Y-%m-%d_%H:%M:%S_%f')),
                                   help="Specifies the output-directory for all emerging "
                                        "files, such as logging and results.")
        smac_opts.add_argument("--input-psmac-dirs", "--input_psmac_dirs", dest='input_psmac_dirs',
                               default=None,
                               help="For parallel SMAC, multiple output-directories "
                                    "are used.")  # TODO: type (list of strings? --> str, nargs=*)
        smac_opts.add_argument("--shared-model", "--shared_model", dest='shared_model',
                               default=False, type=truthy,
                               help="Whether to run SMAC in parallel mode.")
        smac_opts.add_argument("--random-configuration-chooser", "--random_configuration_chooser",
                               dest="random_configuration_chooser",
                               default=None, type=FileType('r'),
                               action=ParseRandomConfigurationChooserAction,
                               help="[dev] path to a python module containing a class"
                                    "`RandomConfigurationChooserImpl` implementing"
                                    "the interface of `RandomConfigurationChooser`")
        smac_opts.add_argument("--hydra-iterations", "--hydra_iterations", dest="hydra_iterations",
                               default=3, type=int,
                               help="[dev] number of hydra iterations. Only active if mode is set to Hydra")
        smac_opts.add_argument("--use-ta-time", "--use_ta_time", dest="use_ta_time",
                               default=False, type=truthy,
                               help="[dev] Instead of measuring SMAC's wallclock time, "
                               "only consider time reported by the target algorithm (ta).")

        # Hyperparameters
        smac_opts.add_argument("--always-race-default", "--always_race_default", dest='always_race_default',
                               default=False, type=truthy,
                               help="[dev] Race new incumbents always against default "
                                    "configuration.")
        smac_opts.add_argument("--intensification-percentage", "--intensification_percentage",
                               dest='intensification_percentage',
                               default=0.5, type=float,
                               help="[dev] The fraction of time to be used on "
                                    "intensification (versus choice of next "
                                    "Configurations).")
        smac_opts.add_argument("--transform_y", "--transform-y",
                               dest='transform_y',
                               choices=["NONE", "LOG", "LOGS", "INVS"],
                               default="NONE",                      
                               help="[dev] Transform all observed cost values"
                               " via log-transformations or inverse scaling."
                               " The subfix \"s\" indicates that SMAC scales the"
                               " y-values accordingly to apply the transformation.")

        ## RF Hyperparameters
        smac_opts.add_argument("--rf_num_trees","--rf-num-trees",
                               dest='rf_num_trees',
                               default=N_TREES, type=int,
                               help="[dev] Number of trees in the random forest (> 1).")
        smac_opts.add_argument("--rf_do_bootstrapping","--rf-do-bootstrapping",
                       dest='rf_do_bootstrapping',
                       default=True, type=bool,
                       help="[dev] Use bootstraping in random forest.")
        smac_opts.add_argument("--rf_ratio_features","--rf-ratio-features",
                       dest='rf_ratio_features',
                       default=5. / 6., type=float,
                       help="[dev] Ratio of sampled features in each split ([0.,1.]).")
        smac_opts.add_argument("--rf_min_samples_split","--rf-min-samples-split",
                       dest='rf_min_samples_split',
                       default=3, type=int,
                       help="[dev] Minimum number of samples to split for building a tree in the random forest.")
        smac_opts.add_argument("--rf_min_samples_leaf","--rf-min-samples-leaf",
                       dest='rf_min_samples_leaf',
                       default=3, type=int,
                       help="[dev] Minimum required number of samples in each leaf of a tree in the random forest.")
        smac_opts.add_argument("--rf_max_depth","--rf-max-depth",
                       dest='rf_max_depth',
                       default=20, type=int,
                       help="[dev] Maximum depth of each tree in the random forest.")
        ## AcquisitionOptimizer SLS
        smac_opts.add_argument("--sls_n_steps_plateau_walk","--sls-n-steps-plateau-walk",
               dest='sls_n_steps_plateau_walk',
               default=10, type=int,
               help="[dev] Maximum number of steps on plateaus during "
                    "the optimization of the acquisition function.")
        smac_opts.add_argument("--sls_max_steps","--sls-max-steps",
               dest='sls_max_steps',
               default=None, type=int,
               help="[dev] Maximum number of local search steps in one iteration"
                    " during the optimization of the acquisition function.")
        smac_opts.add_argument("--acq_opt_challengers","--acq-opt-challengers",
               dest='acq_opt_challengers',
               default=5000, type=int,
               help="[dev] Number of challengers returned by acquisition function" 
                    " optimization. Also influences the number of randomly sampled"
                    " configurations to optimized the acquisition function")
    
        ## Intensification
        smac_opts.add_argument("--intens_adaptive_capping_slackfactor","--intens-adaptive-capping-slackfactork",
               dest='intens_adaptive_capping_slackfactor',
               default=1.2, type=float,
               help="[dev] Slack factor of adpative capping (factor * adpative cutoff)."
                    " Only active if obj is runtime."
                    " If set to very large number it practically deactivates adaptive capping.")
        smac_opts.add_argument("--intens_min_chall","--intens-min-chall",
               dest='intens_min_chall',
               default=2, type=int,
               help="[dev] Minimal number of challengers to be considered in each intensification run (> 1)."
                    " Set to 1 and in combination with very small intensification-percentage."
                    " it will deactivate randomly sampled configurations"
                    " (and hence, extrapolation of random forest will be an issue.)")
        smac_opts.add_argument("--rand_prob","--rand-prob",
               dest='rand_prob',
               default=0.5, type=float,
               help="[dev] probablity to run a random configuration"
               " instead of configuration optimized on the acquisition function")

        self.parser.add_parser(self.smac_parser)
        self.smac_cmd_actions, self.smac_cmd_translations = CMDReader._extract_action_info(self.smac_parser._actions)

    def _add_scen_options(self):
        """Add Scenario Options"""
        self.scen_parser = SMACArgumentParser(formatter_class=ConfigurableHelpFormatter, add_help=False)
        scen_opts = self.scen_parser.add_argument_group("Scenario Options")
        scen_opts.add_argument("--algo", "--ta", dest='ta',
                               type=shlex.split,
                               help="[dev] Specifies the target algorithm call that *SMAC* "
                                    "will optimize. Interpreted as a bash-command.")
        scen_opts.add_argument("--execdir", dest="execdir",
                               default='.', type=str,
                               help="[dev] Specifies the path to the execution-directory.")
        scen_opts.add_argument("--deterministic", dest="deterministic",
                               default=False, type=truthy,
                               help="[dev] If true, the optimization process will be "
                                    "repeatable.")
        scen_opts.add_argument("--run-obj", "--run_obj", dest="run_obj",
                               type=str, action=ProcessRunObjectiveAction,
                               required=True, choices=['runtime', 'quality'],
                               help="[dev] Defines what metric to optimize. When "
                                    "optimizing runtime, *cutoff_time* is "
                                    "required as well.")
        self.overall_obj_arg = \
            scen_opts.add_argument("--overall-obj", "--overall_obj", dest="overall_obj",
                                   type=str, action=ParseOverallObjectiveAction, default='par10',
                                   help="[dev] PARX, where X is an integer defining the "
                                        "penalty imposed on timeouts (i.e. runtimes that "
                                        "exceed the *cutoff-time*).")
        scen_opts.add_argument("--par-factor", "--par_factor", dest="par_factor",
                               type=float, default=10.0,
                               help=SUPPRESS)  # added after parsing --overall-obj
        scen_opts.add_argument("--cost-for-crash", "--cost_for_crash", dest="cost_for_crash",
                               default=float(MAXINT), type=float,
                               help="[dev] Defines the cost-value for crashed runs "
                                    "on scenarios with quality as run-obj.")
        scen_opts.add_argument("--cutoff-time", "--cutoff_time", "--cutoff", dest="cutoff",
                               default=None, type=float,
                               help="[dev] Maximum runtime, after which the "
                                    "target algorithm is cancelled. **Required "
                                    "if *run_obj* is runtime.**")
        scen_opts.add_argument("--memory-limit", "--memory_limit", dest="memory_limit",
                               type=float,
                               help="[dev] Maximum available memory the target algorithm "
                                    "can occupy before being cancelled in MB.")
        scen_opts.add_argument("--tuner-timeout", "--tuner_timeout", "--algo-runs-timelimit", "--algo_runs_timelimit", dest="algo_runs_timelimit",
                               default=float('inf'), type=float,
                               help="[dev] Maximum amount of CPU-time used for optimization.")
        scen_opts.add_argument("--wallclock-limit", "--wallclock_limit", dest="wallclock_limit",
                               default=float('inf'), type=float,
                               help="[dev] Maximum amount of wallclock-time used for optimization.")
        scen_opts.add_argument("--always-race-default", "--always_race_default", dest="always_race_default",
                               default=False, type=truthy,
                               help="[dev] Race new incumbents always against default configuration.")
        scen_opts.add_argument("--runcount-limit", "--runcount_limit", "--ta-run-limit", "--ta_run_limit", dest="ta_run_limit",
                               default=float('inf'), type=float,
                               help="[dev] Maximum number of algorithm-calls during optimization.")
        scen_opts.add_argument("--instance-file", "--instance_file", "--train-inst-fn", "--train_inst_fn", dest='train_inst_fn',
                               type=str, action=ReadTrainInstFileAction,
                               help="[dev] Specifies the file with the training-instances.")
        scen_opts.add_argument("--instances", "--train-insts", "--train_insts", dest="train_insts",
                               default=[[None]],  # overridden by --instance-file
                               help=SUPPRESS)
        scen_opts.add_argument("--test-instance-file", "--test_instance_file", "--test-inst-fn", "--test_inst_fn", dest='test_inst_fn',
                               type=str, action=ReadTestInstFileAction,
                               help="[dev] Specifies the file with the test-instances.")
        scen_opts.add_argument("--test-instances", "--test_instances", "--test-insts", "--test_insts", dest="test_insts",
                               default=[[None]],  # overridden by --test-instance-file
                               help=SUPPRESS)
        scen_opts.add_argument("--feature-file", "--feature_file", "--feature-fn", "--feature_fn", dest='feature_fn',
                               type=str, action=ReadFeatureFileAction,
                               help="[dev] Specifies the file with the instance-features.")
        scen_opts.add_argument("--features", "--feature-dict", "--feature_dict", dest='feature_dict',
                               default={},  # instance name -> feature vector, overridden by --feature-file
                               help=SUPPRESS)
        scen_opts.add_argument("--feature-names", "--feature_names", dest="feature_names",
                               type=list,
                               help=SUPPRESS)  # added after parsing --features
        scen_opts.add_argument("--initial-incumbent", "--initial_incumbent", dest='initial_incumbent',
                               default="DEFAULT", type=str, choices=['DEFAULT', 'RANDOM', 'LHD', 'SOBOL', 'FACTORIAL'],
                               help="[dev] DEFAULT is the default from the PCS.")
        scen_opts.add_argument("--paramfile", "--param-file", "--param_file", "--pcs-fn", "--pcs_fn", dest='pcs_fn',
                               type=str, action=ReadPCSFileAction,
                               help="[dev] Specifies the path to the "
                                    "PCS-file.")
        scen_opts.add_argument('--cs',
                               default=None,  # ConfigSpace object, overridden by --paramfile
                               help=SUPPRESS)

        self.parser.add_parser(self.scen_parser)
        self.scen_cmd_actions, self.scen_cmd_translations = CMDReader._extract_action_info(self.scen_parser._actions)

    def parse_main_command(self, main_cmd_opts: typing.List[str]):
        """Parse main options"""
        args_, misc = self.parser.parse_known_args(main_cmd_opts)
        try:
            misc.remove(self.parser.prog)
        except ValueError:
            pass
        return args_, misc

    def parse_smac_command(self, smac_dict: dict = {}, smac_cmd_opts: typing.List[str] = []):
        """Parse SMAC options"""
        # transform smac dict to smac_args
        try:
            smac_cmd_opts.remove(self.parser.prog)
        except ValueError:
            pass
        smac_cmd = []
        misc_dict = {}
        parsed_smac_args = {}
        for k, v in smac_dict.items():
            if k in self.smac_cmd_translations:
                if not isinstance(v, (str, bool, int, float,)):
                    parsed_smac_args[self.smac_cmd_translations[k]] = v
                else:
                    smac_cmd.append('--' + k.replace('_', '-'))
                    smac_cmd.append(v)
            else:
                misc_dict[k] = v
        smac_cmd.extend(smac_cmd_opts)

        args_, misc_cmd = self.smac_parser.parse_known_args([str(e) for e in smac_cmd])

        # execute output_dir action for default value
        if args_.output_dir == self.output_dir_arg.default:
            self.output_dir_arg(self.parser, args_, self.output_dir_arg.default)

        if args_.shared_model and args_.input_psmac_dirs is None:
            # per default, we assume that
            # all pSMAC runs write to the default output dir
            args_.input_psmac_dirs = "smac3-output*/run*/"

        # copy parsed smac args to smac_args_
        for k, v in parsed_smac_args.items():
            setattr(args_, k, v)

        return args_, misc_dict, misc_cmd

    def parse_scenario_command(self,
                               scenario_file: str=None,
                               scenario_dict: dict={},
                               scenario_cmd_opts: typing.List[str]=[]):
        """
        Parse scenario options
        :param scenario_file: str or None
        Path to the scenario file.
        :param scenario_dict: dict
        Mappings of scenario options to values
        :param scenario_cmd_opts: list[str]
        Scenario options from command line
        :return:
        Parsed scenario arguments
        """
        # read scenario file
        scenario_file_dict = {}
        if isinstance(scenario_file, str):
            scenario_file_dict = in_reader.read_scenario_file(scenario_file)
        elif scenario_file is None:
            pass
        else:
            raise ValueError("Scenario has to be a string or a dictionary")
        # add options from scenario dict (with overwriting)
        scenario_file_dict.update(scenario_dict)
        # transform scenario dict to scen_args
        scen_dict = scenario_file_dict
        scen_cmd = []
        misc_dict = {}
        self.parsed_scen_args.clear()
        for k, v in scen_dict.items():
            if k in self.scen_cmd_translations:
                if not isinstance(v, (str, bool, int, float,)):
                    # e.g. train_insts, test_insts, cs, features
                    self.parsed_scen_args[self.scen_cmd_translations[k]] = v
                else:
                    scen_cmd.append('--' + k.replace('_', '-'))
                    scen_cmd.append(str(v))
            else:
                misc_dict[k] = v
        scen_cmd.extend(scenario_cmd_opts)

        if misc_dict.keys():
            self.logger.warning('Adding unsupported scenario options: {}'.format(misc_dict))
            for k, v in misc_dict.items():
                self.parsed_scen_args[k] = v
            # Fail in a later version:
            # self.scen_parser.exit(1, 'Error: Unknown arguments: {}'.format(misc_dict))

        # append rest of arguments (= override) options from scenario file and
        # parse them with the scenario parser
        scen_args_, misc = self.scen_parser.parse_known_args([str(e) for e in scen_cmd])

        if misc:
            self.scen_parser.exit(1, 'Error: Can not parse arguments: {}'.format(misc))

        # execute overall_obj action for default value
        if scen_args_.overall_obj == self.overall_obj_arg.default:
            self.overall_obj_arg(self.scen_parser, scen_args_, self.overall_obj_arg.default)

        # make checks that argparse can't perform natively

        if "cutoff_time_required" in self.parsed_scen_args:
            if self.parsed_scen_args["cutoff_time_required"] and not scen_args_.cutoff:
                self.parser.exit(1, "Error: {}".format(self.parsed_scen_args["cutoff_time_required"]["error"]))
            self.parsed_scen_args.pop("cutoff_time_required")

        # copy parsed scenario args to scen_args_
        # par_factor, train_insts, test_insts, (features, features_names), feature_dict, cs
        for k, v in self.parsed_scen_args.items():
            setattr(scen_args_, k, v)

        return scen_args_

    def read_smac_scenario_dict_cmd(self, dict_cmd: dict, scenario_file: str=None):
        """Reads smac and scenario options provided in a dictionary

        Returns
        -------
            smac_args_, scen_args_: smac and scenario options parsed with corresponding ArgumentParser
        """
        smac_args_, misc_dict, misc_cmd = self.parse_smac_command(smac_dict=dict_cmd)
        scen_args_ = self.parse_scenario_command(scenario_file=scenario_file,
                                                 scenario_dict=misc_dict,
                                                 scenario_cmd_opts=misc_cmd)
        return smac_args_, scen_args_

    def read_cmd(self, commandline_arguments: typing.List[str]=sys.argv[1:]):
        """Reads command line options (main, smac and scenario options)

        Returns
        -------
            smac_args_, scen_args_: parsed arguments;
                main, smac and scenario options parsed with corresponding ArgumentParser
        """
        main_args_, misc = self.parse_main_command(main_cmd_opts=commandline_arguments)
        smac_args_, misc_dict, misc_cmd = self.parse_smac_command(smac_cmd_opts=misc)
        scen_args_ = self.parse_scenario_command(scenario_file=main_args_.scenario_file,
                                                 scenario_dict=misc_dict,
                                                 scenario_cmd_opts=misc_cmd)
        return main_args_, smac_args_, scen_args_

    @staticmethod
    def _write_options_to_doc(_arguments: dict, path: str, exclude: typing.List[str]):
        with open(path, 'w') as fh:
            for arg in sorted(_arguments.keys()):
                print_arg = arg.lstrip('-').replace('-', '_')
                if print_arg in exclude:
                    continue
                if _arguments[arg]['help'] == SUPPRESS:
                    continue
                fh.write(":{}: ".format(print_arg))
                fh.write("{}".format(_arguments[arg]['help'].lstrip("[dev] ")))
                if 'default' in _arguments[arg] and _arguments[arg]['default']:
                    fh.write(" Default: {}.".format(_arguments[arg]['default']))
                if 'choice' in _arguments[arg] and _arguments[arg]['choice']:
                    fh.write(" Must be from: {}.".format(_arguments[arg]['choice']))
                fh.write("\n")
            fh.write("\n\n")

    def write_main_options_to_doc(self, path: str='main_options.rst'):
        """Writes the SMAC option-list to file for autogeneration in documentation.
        The list is created in doc/conf.py and read in doc/options.rst.

        Parameters
        ----------
        path: string
            Where to write to (relative to doc-folder since executed in conf.py)
        """
        exclude = []
        _arguments = self.main_cmd_actions
        CMDReader._write_options_to_doc(_arguments, path, exclude)

    def write_smac_options_to_doc(self, path: str='smac_options.rst'):
        """Writes the SMAC option-list to file for autogeneration in documentation.
        The list is created in doc/conf.py and read in doc/options.rst.

        Parameters
        ----------
        path: string
            Where to write to (relative to doc-folder since executed in conf.py)
        """
        exclude = []
        _arguments = self.smac_cmd_actions
        CMDReader._write_options_to_doc(_arguments, path, exclude)

    def write_scenario_options_to_doc(self, path: str='scenario_options.rst'):
        """Writes the Scenario option-list to file for autogeneration in documentation.
        The list is created in doc/conf.py and read in doc/options.rst.

        Parameters
        ----------
        path: string
            Where to write to (relative to doc-folder since executed in conf.py)
        """
        exclude = ['cs', 'features', 'instances', 'test_instances']
        _arguments = self.scen_cmd_actions
        CMDReader._write_options_to_doc(_arguments, path, exclude)
