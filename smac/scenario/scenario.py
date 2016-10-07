from collections import defaultdict
import os
import sys
import logging
import numpy
import shlex
import time
import datetime

from smac.utils.io.input_reader import InputReader
from smac.configspace import pcs

__author__ = "Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.2"


class Scenario(object):

    '''
    main class of SMAC
    '''

    def __init__(self, scenario, cmd_args=None):
        """Construct scenario object from file or dictionary.

        Parameters
        ----------
        scenario : str or dict
            if str, it will be interpreted as to a path a scenario file
            if dict, it will be directly to get all scenario related information
        cmd_args : dict
            command line arguments that were not processed by argparse

        """
        self.logger = logging.getLogger("scenario")
        self.in_reader = InputReader()

        if type(scenario) is str:
            scenario_fn = scenario
            self.logger.info("Reading scenario file: %s" % (scenario_fn))
            scenario = self.in_reader.read_scenario_file(scenario_fn)
        elif type(scenario) is dict:
            pass
        else:
            raise TypeError(
                "Wrong type of scenario (str or dict are supported)")

        if cmd_args:
            scenario.update(cmd_args)

        self._arguments = {}
        self._groups = defaultdict(set)
        self._add_arguments()

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

    def add_argument(self, name, help, callback=None, default=None,
                     dest=None, required=False, mutually_exclusive_group=None):
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
        """
        if not isinstance(required, bool):
            raise TypeError("Argument 'required' must be of type 'bool'.")
        if required is not False and mutually_exclusive_group is not None:
            raise ValueError("Cannot make argument '%s' required and add it to"
                             " a group of mutually exclusive arguments." % name)

        self._arguments[name] = {'default': default,
                                 'required': required,
                                 'help': help,
                                 'dest': dest,
                                 'callback': callback}

        if mutually_exclusive_group:
            self._groups[mutually_exclusive_group].add(name)


    def _parse_argument(self, name, scenario, help, callback=None, default=None,
                        dest=None, required=False):
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

        return dest, value

    def _add_arguments(self):
        # Add allowed arguments
        self.add_argument(name='algo', help=None, dest='ta',
                          callback=lambda arg: shlex.split(arg))
        self.add_argument(name='execdir', default='.', help=None)
        self.add_argument(name='deterministic', default="0", help=None,
                          callback=lambda arg: arg in ["1", "true", True])
        self.add_argument(name='paramfile', help=None, dest='pcs_fn',
                          mutually_exclusive_group='cs')
        self.add_argument(name='run_obj', help=None, default='runtime')
        self.add_argument(name='overall_obj', help=None, default='par10')
        self.add_argument(name='cutoff_time', help=None, default=None,
                          dest='cutoff', callback=lambda arg: float(arg))
        self.add_argument(name='memory_limit', help=None)
        self.add_argument(name='tuner-timeout', help=None, default=numpy.inf,
                          dest='algo_runs_timelimit',
                          callback=lambda arg: float(arg))
        self.add_argument(name='wallclock_limit', help=None, default=numpy.inf,
                          callback=lambda arg: float(arg))
        self.add_argument(name='runcount_limit', help=None, default=numpy.inf,
                          callback=lambda arg: float(arg), dest="ta_run_limit")
        self.add_argument(name='instance_file', help=None, dest='train_inst_fn')
        self.add_argument(name='test_instance_file', help=None,
                          dest='test_inst_fn')
        self.add_argument(name='feature_file', help=None, dest='feature_fn')
        self.add_argument(name='output_dir', help=None,
                          default="smac3-output_%s" % (
                              datetime.datetime.fromtimestamp(
                                  time.time()).strftime(
                                  '%Y-%m-%d_%H:%M:%S')))
        self.add_argument(name='shared_model', help=None, default='0',
                          callback=lambda arg: arg in ['1', 'true', True])
        self.add_argument(name='instances', default=[[None]], help=None,
                          dest='train_insts')
        self.add_argument(name='test_instances', default=[[None]], help=None,
                          dest='test_insts')
        # instance name -> feature vector
        self.add_argument(name='features', default={}, help=None,
                          dest='feature_dict')
        # ConfigSpace object
        self.add_argument(name='cs', help=None, mutually_exclusive_group='cs')

    def _transform_arguments(self):
        self.n_features = len(self.feature_dict)
        self.feature_array = None

        if self.overall_obj[:3] in ["PAR", "par"]:
            self.par_factor = int(self.overall_obj[3:])
        elif self.overall_obj[:4] in ["mean", "MEAN"]:
            self.par_factor = int(self.overall_obj[4:])
        else:
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

        # read feature file
        if self.feature_fn:
            if os.path.isfile(self.feature_fn):
                self.feature_dict = self.in_reader.read_instance_features_file(
                    self.feature_fn)[1]

        if self.feature_dict:
            self.n_features = len(
                self.feature_dict[list(self.feature_dict.keys())[0]])
            self.feature_array = []
            for inst_ in self.train_insts:
                self.feature_array.append(self.feature_dict[inst_])
            self.feature_array = numpy.array(self.feature_array)

        # read pcs file
        if self.pcs_fn and os.path.isfile(self.pcs_fn):
            with open(self.pcs_fn) as fp:
                pcs_str = fp.readlines()
                self.cs = pcs.read(pcs_str)
                self.cs.seed(42)
        elif self.pcs_fn:
            self.logger.error("Have not found pcs file: %s" %
                              (self.pcs_fn))
            sys.exit(1)

        self.logger.info("Output to %s" % (self.output_dir))


