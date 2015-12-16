import os
import sys
import logging
import numpy

from smac.utils.io.input_reader import InputReader
from smac.configspace import pcs

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class Scenario(object):
    '''
    main class of SMAC
    '''

    def __init__(self, args_):
        '''
            constructor
            Attributes
            ----------
                args_: ArgumentParse object
                    all command line options
        '''
        self.logger = logging.getLogger("scenario")

        self.args_ = args_

        self.logger.info("Reading scenario file: %s" % (args_.scenario_file))
        in_reader = InputReader()
        scenario = in_reader.read_scenario_file(args_.scenario_file)

        self.ta = scenario["algo"]
        self.execdir = scenario.get("execdir", ".")
        self.deterministic = scenario.get("deterministic", "0")
        self.pcs_fn = scenario["paramfile"]
        self.run_obj = scenario.get("run_obj", "runtime")
        self.overall_obj = scenario.get("overall_obj", "par10")
        self.cutoff = scenario.get("cutoff_time", 999999999)
        self.algo_runs_timelimit = scenario.get("tunerTimeout", numpy.inf)
        self.wallclock_limit = scenario.get("wallclock-limit", numpy.inf)
        self.train_inst_fn = scenario.get("instance_file", None)
        self.test_inst_fn = scenario.get("test_instance_file", None)
        self.feature_fn = scenario.get("feature_file")
        # not handled: outdir (and some more)

        self.train_insts = []
        self.test_ints = []
        self.feature_dict = {}  # instance name -> feature vector
        self.feature_array = None
        self.cs = None  # ConfigSpace object

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
                self.test_ints = in_reader.read_instance_file(
                    self.test_inst_fn)
            else:
                self.logger.error(
                    "Have not found test instance file: %s" % (self.test_inst_fn))
                sys.exit(1)
        if self.feature_fn:
            if os.path.isfile(self.feature_fn):
                self.feature_dict = in_reader.read_instance_features_file(
                    self.feature_fn)[1]
                self.feature_array = []
                for inst_ in self.train_insts:
                    self.feature_array.append(self.feature_dict[inst_[0]])
                self.feature_array = numpy.array(self.feature_array) 

        if os.path.isfile(self.pcs_fn):
            with open(self.pcs_fn) as fp:
                pcs_str = fp.readlines()
                # TODO: ConfigSpace should use only logging and no print
                # statements
                self.cs = pcs.read(pcs_str)
        else:
            self.logger.error("Have not found pcs file: %s" %
                              (self.pcs_fn))
            sys.exit(1)
