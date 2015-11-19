import os
import sys
import logging
from smac.configspace import pcs

from utils.io.cmd_reader import CMDReader
from utils.io.input_reader import InputReader

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class SMAC(object):
    '''
    main class of SMAC
    '''

    def __init__(self):
        '''
            constructor
        '''
        self.logger = logging.getLogger("SMAC")

    def main_cli(self):
        '''
            main function of SMAC for CLI interface
        '''

        cmd_reader = CMDReader()
        args_ = cmd_reader.read_cmd()

        self.logger.info("Reading scenario file: %s" % (args_.scenario_file))
        in_reader = InputReader()
        scenario = in_reader.read_scenario_file(args_.scenario_file)

        if scenario.get("instance_file"):
            if os.path.isfile(scenario["instance_file"]):
                training_insts = in_reader.read_instance_file(
                    scenario["instance_file"])
            else:
                self.logger.error(
                    "Have not found instance file: %s" % (scenario["instance_file"]))
                sys.exit(1)
        if scenario.get("test_instance_file"):
            if os.path.isfile(scenario["test_instance_file"]):
                test_insts = in_reader.read_instance_file(
                    scenario["test_instance_file"])
            else:
                self.logger.error(
                    "Have not found test instance file: %s" % (scenario["test_instance_file"]))
                sys.exit(1)
        if scenario.get("feature_file"):
            if os.path.isfile(scenario["feature_file"]):
                inst_features = in_reader.read_instance_features_file(
                    scenario["feature_file"])
            else:
                self.logger.error(
                    "Have not found test instance file: %s" % (scenario["feature_file"]))
                sys.exit(1)

        cs = pcs(scenario["paramfile"])
        #pcs = read_pcs(scenario["paramfile"])
