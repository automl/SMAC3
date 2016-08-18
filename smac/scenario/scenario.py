import os
import sys
import logging
import numpy
import shlex
import time
import datetime

from smac.utils.io.input_reader import InputReader
from smac.configspace import pcs

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


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

        if type(scenario) is str:
            scenario_fn = scenario
            self.logger.info("Reading scenario file: %s" % (scenario_fn))
            in_reader = InputReader()
            scenario = in_reader.read_scenario_file(scenario_fn)
        elif type(scenario) is dict:
            in_reader = InputReader()
        else:
            raise TypeError(
                "Wrong type of scenario (str or dict are supported)")

        if cmd_args:
            scenario.update(cmd_args)

        self.ta = shlex.split(scenario.get("algo", ""))
        self.execdir = scenario.get("execdir", ".")
        self.deterministic = scenario.get("deterministic", "0") == "1" \
            or scenario.get("deterministic", "0") == "true" \
            or scenario.get('deterministic', '0') is True
        self.pcs_fn = scenario.get("paramfile", None)
        self.run_obj = scenario.get("run_obj", "runtime")
        self.overall_obj = scenario.get("overall_obj", "par10")
        self.cutoff = float(scenario.get("cutoff_time", 999999999))
        self.algo_runs_timelimit = float(
            scenario.get("tunerTimeout", numpy.inf))
        self.wallclock_limit = float(
            scenario.get("wallclock-limit", numpy.inf))
        self.ta_run_limit = float(scenario.get("runcount-limit", numpy.inf))
        self.train_inst_fn = scenario.get("instance_file", None)
        self.test_inst_fn = scenario.get("test_instance_file", None)
        self.feature_fn = scenario.get("feature_file")
        self.output_dir = scenario.get("output_dir", "smac3-output_%s" % (
            datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')))
        self.logger.info("Output to %s" %(self.output_dir))
        self.shared_model = scenario.get("shared_model", "0") == "1" \
            or scenario.get("shared_model", "0") == "true" \
            or scenario.get('shared_model', '0') is True

        self.train_insts = scenario.get("instances", [[None]])
        self.test_insts = scenario.get("test_instances", [])
        # instance name -> feature vector
        self.feature_dict = scenario.get("features", {})
        self.n_features = len(self.feature_dict)
        self.feature_array = None
        self.cs = scenario.get("cs", None)  # ConfigSpace object

        if self.overall_obj[:3] in ["PAR", "par"]:
            self.par_factor = int(self.overall_obj[3:])
        elif self.overall_obj[:4] in ["mean", "MEAN"]:
            self.par_factor = int(self.overall_obj[4:])
        else:
            self.par_factor = 1

        # read instance files
        if self.train_inst_fn:
            if os.path.isfile(self.train_inst_fn):
                self.train_insts = in_reader.read_instance_file(
                    self.train_inst_fn)
            else:
                self.logger.error(
                    "Have not found instance file: %s" % (self.train_inst_fn))
                sys.exit(1)
        if self.test_inst_fn:
            if os.path.isfile(self.test_inst_fn):
                self.test_insts = in_reader.read_instance_file(
                    self.test_inst_fn)
            else:
                self.logger.error(
                    "Have not found test instance file: %s" % (self.test_inst_fn))
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
                self.feature_dict = in_reader.read_instance_features_file(
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
