import logging
import copy
import typing

import numpy as np

from smac.utils.io.input_reader import InputReader
from smac.utils.io.output_writer import OutputWriter
from smac.utils.io.cmd_reader import CMDReader


__author__ = "Marius Lindauer, Matthias Feurer, Aaron Kimmig"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.2"


class Scenario(object):

    """
    Scenario contains the configuration of the optimization process and
    constructs a scenario object from a file or dictionary.

    All arguments set in the Scenario are set as attributes.

    """

    def __init__(self, scenario=None, cmd_options: dict=None):
        """ Creates a scenario-object. The output_dir will be
        "output_dir/run_id/" and if that exists already, the old folder and its
        content will be moved (without any checks on whether it's still used by
        another process) to "output_dir/run_id.OLD". If that exists, ".OLD"s
        will be appended until possible.

        Parameters
        ----------
        scenario : str or dict or None
            If str, it will be interpreted as to a path a scenario file
            If dict, it will be directly to get all scenario related information
            If None, only cmd_options will be used
        cmd_options : dict
            Options from parsed command line arguments
        """
        self.logger = logging.getLogger(
            self.__module__ + '.' + self.__class__.__name__)
        self.PCA_DIM = 7

        self.in_reader = InputReader()
        self.out_writer = OutputWriter()

        self.output_dir_for_this_run = None

        self._arguments = {}
        self._arguments.update(CMDReader().scen_cmd_actions)

        if scenario is None:
            scenario = {}
        if isinstance(scenario, str):
            scenario_fn = scenario
            scenario = {}
            if cmd_options:
                scenario.update(cmd_options)
            cmd_reader = CMDReader()
            self.logger.info("Reading scenario file: %s", scenario_fn)
            smac_args_, scen_args_ = cmd_reader.read_smac_scenario_dict_cmd(scenario, scenario_fn)
            scenario = {}
            scenario.update(vars(smac_args_))
            scenario.update(vars(scen_args_))
        elif isinstance(scenario, dict):
            scenario = copy.copy(scenario)
            if cmd_options:
                scenario.update(cmd_options)
            cmd_reader = CMDReader()
            smac_args_, scen_args_ = cmd_reader.read_smac_scenario_dict_cmd(scenario)
            scenario = {}
            scenario.update(vars(smac_args_))
            scenario.update(vars(scen_args_))
        else:
            raise TypeError(
                "Wrong type of scenario (str or dict are supported)")


        for arg_name, arg_value in scenario.items():
            setattr(self, arg_name, arg_value)

        self._transform_arguments()

        self.logger.debug("SMAC and Scenario Options:")
        if cmd_options:
            for arg_name, arg_value in cmd_options.items():
                if isinstance(arg_value, (int, str, float)):
                    self.logger.debug("%s = %s" %(arg_name, arg_value))

    def _transform_arguments(self):
        """TODO"""

        self.n_features = len(self.feature_dict)
        self.feature_array = None

        self.instance_specific = {}
        
        if self.run_obj == "runtime":
            self.logy = True

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

        if self.feature_dict:
            self.feature_array = []
            for inst_ in self.train_insts:
                self.feature_array.append(self.feature_dict[inst_])
            self.feature_array = np.array(self.feature_array)
            self.n_features = self.feature_array.shape[1]

        if self.use_ta_time:
            if self.algo_runs_timelimit is None or not np.isfinite(self.algo_runs_timelimit):
                self.algo_runs_timelimit = self.wallclock_limit
            self.wallclock_limit = np.inf

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
            self.logger.warning("All instances were casted to str.")
        return l

    def write(self):
        """ Write scenario to self.output_dir/scenario.txt. """
        self.out_writer.write_scenario_file(self)
