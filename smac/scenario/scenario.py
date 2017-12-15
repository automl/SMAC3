from collections import defaultdict
import os
import sys
import logging
import numpy
import shlex
import time
import datetime
import copy
import typing

from smac.utils.io.input_reader import InputReader
from smac.utils.io.output_writer import OutputWriter
from smac.utils.constants import MAXINT
from smac.configspace import pcs, pcs_new


__author__ = "Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.2"


def _is_truthy(arg):
    if arg in ["1", "true", "True", True]:
        return True
    elif arg in ["0", "false", "False", False]:
        return False
    else:
        raise ValueError("{} cannot be interpreted as a boolean argument. "
                         "Please use one of {{0, false, 1, true}}.".format(arg))


class Scenario(object):

    """
    Scenario contains the configuration of the optimization process and
    constructs a scenario object from a file or dictionary.

    All arguments set in the Scenario are set as attributes.

    """

    def __init__(self, scenario, cmd_args: dict=None):
        """ Creates a scenario-object. The output_dir will be
        "output_dir/run_id/" and if that exists already, the old folder and its
        content will be moved (without any checks on whether it's still used by
        another process) to "output_dir/run_id.OLD". If that exists, ".OLD"s
        will be appended until possible.

        Parameters
        ----------
        scenario : str or dict
            If str, it will be interpreted as to a path a scenario file
            If dict, it will be directly to get all scenario related information
        cmd_args : dict
            Command line arguments that were not processed by argparse
        """
        self.logger = logging.getLogger(
            self.__module__ + '.' + self.__class__.__name__)
        self.PCA_DIM = 7

        self.in_reader = InputReader()
        self.out_writer = OutputWriter()

        self.output_dir_for_this_run = None

        if type(scenario) is str:
            scenario_fn = scenario
            self.logger.info("Reading scenario file: %s" % (scenario_fn))
            scenario = self.in_reader.read_scenario_file(scenario_fn)
        elif type(scenario) is dict:
            scenario = copy.copy(scenario)
        else:
            raise TypeError(
                "Wrong type of scenario (str or dict are supported)")

        if cmd_args:
            scenario.update(cmd_args)

        self._arguments = {}
        self._groups = defaultdict(set)
        self._add_arguments()

        # Make cutoff mandatory if run_obj is runtime
        if scenario['run_obj'] == 'runtime':
            self._arguments['cutoff_time']['required'] = True

        # Parse arguments
        parsed_arguments = {}
        for key, value in self._arguments.items():
            arg_name, arg_value = self._parse_argument(key, scenario, **value)
            parsed_arguments[arg_name] = arg_value

        if len(scenario) != 0:
            raise ValueError('Could not parse the following arguments: %s' %
                             str(list(scenario.keys())))

        for group, potential_members in self._groups.items():
            n_members_in_scenario = 0
            for pm in potential_members:
                if pm in parsed_arguments:
                    n_members_in_scenario += 1

            if n_members_in_scenario != 1:
                raise ValueError('Exactly one of the following arguments must '
                                 'be specified in the scenario file: %s' %
                                 str(potential_members))

        for arg_name, arg_value in parsed_arguments.items():
            setattr(self, arg_name, arg_value)

        self._transform_arguments()

        self.logger.debug("Scenario Options:")
        for arg_name, arg_value in parsed_arguments.items():
            if isinstance(arg_value,(int,str,float)):
                self.logger.debug("%s = %s" %(arg_name,arg_value))

    def add_argument(self, name: str, help: str, callback=None, default=None,
                     dest: str=None, required: bool=False,
                     mutually_exclusive_group: str=None,
                     choice=None):
        """Add argument to the scenario object.

        Parameters
        ----------
        name : str
            Argument name
        help : str
            Help text which can be displayed in the documentation.
        callback : callable, optional
            If given, the callback will be called when the argument is
            parsed. Useful for custom casting/typechecking.
        default : object, optional
            Default value if the argument is not given. Default to ``None``.
        dest : str
            Assign the argument to scenario object by this name.
        required : bool
            If True, the scenario will raise an error if the argument is not
            given.
        mutually_exclusive_group : str
            Group arguments with similar behaviour by assigning the same string
            value. The scenario will ensure that exactly one of the arguments is
            given. Is used for example to ensure that either a configuration
            space object or a parameter file is passed to the scenario. Can not
            be used together with ``required``.
        choice: list/set/tuple
            List of possible string for this argument
        """
        if not isinstance(required, bool):
            raise TypeError("Argument 'required' must be of type 'bool'.")
        if required is not False and mutually_exclusive_group is not None:
            raise ValueError("Cannot make argument '%s' required and add it to"
                             " a group of mutually exclusive arguments." % name)
        if choice is not None and not isinstance(choice, (list, set, tuple)):
            raise TypeError('Choice must be of type list/set/tuple.')

        self._arguments[name] = {'default': default,
                                 'required': required,
                                 'help': help,
                                 'dest': dest,
                                 'callback': callback,
                                 'choice': choice}

        if mutually_exclusive_group:
            self._groups[mutually_exclusive_group].add(name)

    def _parse_argument(self, name: str, scenario: dict, help: str,
                        callback=None, default=None,
                        dest: str=None, required: bool=False, choice=None):
        """Search the scenario dict for a single allowed argument and parse it.

        Side effect: the argument is removed from the scenario dict if found.

        name : str
            Argument name, as specified in the Scenario class.
        scenario : dict
            Scenario dict as provided by the user or as parsed by the cli
            interface.
        help : str
            Help string of the argument
        callback : callable, optional (default=None)
            If given, will be called to transform the given argument.
        default : object, optional (default=None)
            Will be used as default value if the argument is not given by the
            user.
        dest : str, optional (default=None)
            Will be used as member name of the scenario.
        required : bool (default=False)
            If ``True``, the scenario will raise an Exception if the argument is
            not given.
        choice : list, optional (default=None)
            If given, the scenario checks whether the argument is in the
            list. If not, it raises an Exception.

        Returns
        -------
        str
            Member name of the attribute.
        object
            Value of the attribute.
        """
        normalized_name = name.lower().replace('-', '').replace('_', '')
        value = None

        # Allows us to pop elements in order to remove all parsed elements
        # from the dictionary
        for key in list(scenario.keys()):
            # Check all possible ways to spell an argument
            normalized_key = key.lower().replace('-', '').replace('_', '')
            if normalized_key == normalized_name:
                value = scenario.pop(key)

        if dest is None:
            dest = name.lower().replace('-', '_')

        if required is True:
            if value is None:
                raise ValueError('Required scenario argument %s not given.' %
                                 name)

        if value is None:
            value = default

        if value is not None and callable(callback):
            value = callback(value)

        if value is not None and choice:
            value = value.strip()
            if value not in choice:
                raise ValueError('Argument %s can only take a value in %s, '
                                 'but is %s' % (name, choice, value))

        return dest, value

    def _add_arguments(self):
        """TODO"""
        # Add allowed arguments
        self.add_argument(name='abort_on_first_run_crash',
                          help="If true, *SMAC* will abort if the first run of "
                               "the target algorithm crashes.",
                          default=True, callback=_is_truthy)
        self.add_argument(name='always_race_default', default=False,
                          help="Race new incumbents always against default "
                               "configuration.",
                          callback=_is_truthy, dest="always_race_default")
        self.add_argument(name='algo', dest='ta', callback=shlex.split,
                          help="Specifies the target algorithm call that *SMAC* "
                               "will optimize. Interpreted as a bash-command.")
        self.add_argument(name='execdir', default='.',
                          help="Specifies the path to the execution-directory.")
        self.add_argument(name='deterministic', default=False,
                          help="If true, the optimization process will be "
                               "repeatable.", callback=_is_truthy)
        self.add_argument(name='intensification_percentage', default=0.5,
                          help="The fraction of time to be used on "
                               "intensification (versus choice of next "
                                "Configurations).", callback=float)
        self.add_argument(name='paramfile', help="Specifies the path to the "
                                                 "PCS-file.",
                          dest='pcs_fn', mutually_exclusive_group='cs')
        self.add_argument(name='run_obj',
                          help="Defines what metric to optimize. When "
                               "optimizing runtime, *cutoff_time* is "
                               "required as well.",
                          required=True, choice=['runtime', 'quality'])
        self.add_argument(name='overall_obj',
                          help="PARX, where X is an integer defining the "
                               "penalty imposed on timeouts (i.e. runtimes that "
                               "exceed the *cutoff-time*).",
                          default='par10')
        self.add_argument(name='cost_for_crash', default=float(MAXINT),
                          help="Defines the cost-value for crashed runs "
                               "on scenarios with quality as run-obj.",
                          callback=float)
        self.add_argument(name='cutoff_time',
                          help="Maximum runtime, after which the "
                               "target algorithm is cancelled. **Required "
                               "if *run_obj* is runtime.**", default=None,
                          dest='cutoff', callback=float)
        self.add_argument(name='memory_limit',
                          help="Maximum available memory the target algorithm "
                               "can occupy before being cancelled.")
        self.add_argument(name='tuner-timeout',
                          help="Maximum amount of CPU-time used for optimization.",
                          default=numpy.inf,
                          dest='algo_runs_timelimit', callback=float)
        self.add_argument(name='wallclock_limit',
                          help="Maximum amount of wallclock-time used for optimization.",
                          default=numpy.inf, callback=float)
        self.add_argument(name='always_race_default',
                          help="Race new incumbents always against default configuration.",
                          default=False,
                          callback=_is_truthy, dest="always_race_default")
        self.add_argument(name='runcount_limit',
                          help="Maximum number of algorithm-calls during optimization.",
                          default=numpy.inf, callback=float, dest="ta_run_limit")
        self.add_argument(name='minR',
                          help="Minimum number of calls per configuration.",
                          default=1, callback=int, dest='minR')
        self.add_argument(name='maxR',
                          help="Maximum number of calls per configuration.",
                          default=2000, callback=int, dest='maxR')
        self.add_argument(name='instance_file',
                          help="Specifies the file with the training-instances.",
                          dest='train_inst_fn')
        self.add_argument(name='test_instance_file',
                          help="Specifies the file with the test-instances.",
                          dest='test_inst_fn')
        self.add_argument(name='feature_file',
                          help="Specifies the file with the instance-features.",
                          dest='feature_fn')
        self.add_argument(name='output_dir',
                          help="Specifies the output-directory for all emerging "
                               "files, such as logging and results.",
                          default="smac3-output_%s" % (
                              datetime.datetime.fromtimestamp(
                                  time.time()).strftime(
                                  '%Y-%m-%d_%H:%M:%S_%f')))
        self.add_argument(name='input_psmac_dirs', default=None,
                          help="For parallel SMAC, multiple output-directories "
                               "are used.")
        self.add_argument(name='shared_model',
                          help="Whether to run SMAC in parallel mode.",
                          default=False, callback=_is_truthy)
        self.add_argument(name='instances', default=[[None]], help=None,
                          dest='train_insts')
        self.add_argument(name='test_instances', default=[[None]], help=None,
                          dest='test_insts')
        self.add_argument(name='initial_incumbent', default="DEFAULT",
                          help="DEFAULT is the default from the PCS.",
                          dest='initial_incumbent',
                          choice=['DEFAULT', 'RANDOM'])
        # instance name -> feature vector
        self.add_argument(name='features', default={}, help=None,
                          dest='feature_dict')
        # ConfigSpace object
        self.add_argument(name='cs', help=None, mutually_exclusive_group='cs')

    def _transform_arguments(self):
        """TODO"""
        self.n_features = len(self.feature_dict)
        self.feature_array = None

        if self.overall_obj[:3] in ["PAR", "par"]:
            par_str = self.overall_obj[3:]
        elif self.overall_obj[:4] in ["mean", "MEAN"]:
            par_str = self.overall_obj[4:]
        # Check for par-value as in "par10"/ "mean5"
        if len(par_str) > 0:
            self.par_factor = int(par_str)
        else:
            self.logger.debug("No par-factor detected. Using 1 by default.")
            self.par_factor = 1

        # read instance files
        if self.train_inst_fn:
            if os.path.isfile(self.train_inst_fn):
                self.train_insts = self.in_reader.read_instance_file(
                    self.train_inst_fn)
            else:
                self.logger.error(
                    "Have not found instance file: %s" % (self.train_inst_fn))
                sys.exit(1)
        if self.test_inst_fn:
            if os.path.isfile(self.test_inst_fn):
                self.test_insts = self.in_reader.read_instance_file(
                    self.test_inst_fn)
            else:
                self.logger.error(
                    "Have not found test instance file: %s" % (
                        self.test_inst_fn))
                sys.exit(1)

        self.instance_specific = {}

        def extract_instance_specific(instance_list):
            insts = []
            for inst in instance_list:
                if len(inst) > 1:
                    self.instance_specific[inst[0]] = " ".join(inst[1:])
                insts.append(inst[0])
            return insts

        self.train_insts = extract_instance_specific(self.train_insts)
        if self.test_insts:
            self.test_insts = extract_instance_specific(self.test_insts)

        self.train_insts = self._to_str_and_warn(l=self.train_insts)
        self.test_insts = self._to_str_and_warn(l=self.test_insts)

        # read feature file
        if self.feature_fn:
            if os.path.isfile(self.feature_fn):
                features = self.in_reader.read_instance_features_file(
                    self.feature_fn)
                self.feature_names, self.feature_dict = features

        if self.feature_dict:
            self.feature_array = []
            for inst_ in self.train_insts:
                self.feature_array.append(self.feature_dict[inst_])
            self.feature_array = numpy.array(self.feature_array)
            self.n_features = self.feature_array.shape[1]

        # read pcs file
        if self.pcs_fn and os.path.isfile(self.pcs_fn):
            with open(self.pcs_fn) as fp:
                pcs_str = fp.readlines()
                try:
                    self.cs = pcs.read(pcs_str)
                except:
                    self.logger.debug("Could not parse pcs file with old format; trying new format next")
                    self.cs = pcs_new.read(pcs_str)
                self.cs.seed(42)
        elif self.pcs_fn:
            self.logger.error("Have not found pcs file: %s" %
                              (self.pcs_fn))
            sys.exit(1)

        # you cannot set output dir to None directly
        # because None is replaced by default always
        if self.output_dir == "":
            self.output_dir = None
            self.logger.debug("Deactivate output directory.")
        else:
            self.logger.info("Output to %s" % (self.output_dir))

        if self.shared_model and self.input_psmac_dirs is None:
            # per default, we assume that
            # all psmac runs write to the same directory
            self.input_psmac_dirs = [self.output_dir]

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.logger = logging.getLogger(
            self.__module__ + '.' + self.__class__.__name__)

    def _to_str_and_warn(self, l: typing.List[typing.Any]):
        warn_ = False
        for i, e in enumerate(l):
            if e is not None and not isinstance(e, str):
                warn_ = True
                try:
                    l[i] = str(e)
                except ValueError:
                    raise ValueError("Failed to cast all instances to str")
        if warn_:
            self.logger.warn("All instances were casted to str.")
        return l

    def write(self):
        """ Write scenario to self.output_dir/scenario.txt. """
        self.out_writer.write_scenario_file(self)

    def write_options_to_doc(self, path='scenario_options.rst'):
        """Writes the option-list to file for autogeneration in documentation.
        The list is created in doc/conf.py and read in doc/options.rst.

        Parameters
        ----------
        path: string
            Where to write to (relative to doc-folder since executed in conf.py)
        """
        exclude = ['cs', 'features', 'instances', 'test_instances']
        with open(path, 'w') as fh:
            for arg in sorted(self._arguments.keys()):
                if arg in exclude:
                    continue
                fh.write(":{}: ".format(arg))
                fh.write("{}".format(self._arguments[arg]['help']))
                if self._arguments[arg]['default']:
                    fh.write(" Default: {}.".format(self._arguments[arg]['default']))
                if self._arguments[arg]['choice']:
                    fh.write(" Must be from: {}.".format(self._arguments[arg]['choice']))
                fh.write("\n")
            fh.write("\n\n")
