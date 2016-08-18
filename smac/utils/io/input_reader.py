__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"

import numpy as np
from smac.configspace import pcs


class InputReader(object):

    """
        reading all input files for SMAC (scenario file, instance files, ...)

        Note: most of this code was taken from the pysmac repository.
              We copy it here because we don't want smac3 to depend
              on an earlier version!
    """

    def __init__(self):
        """
        Constructor
        """
        pass

    def read_scenario_file(self, fn):
        """
            encapsulates read_scenario_file of pysmac

            Returns
            -------
            dict : dictionary
                (key, value) pairs are (variable name, variable value)
        """
        # translate the difference option names to a canonical name
        scenario_option_names = {'algo-exec': 'algo',
                                 'algoExec': 'algo',
                                 'algo-exec-dir': 'execdir',
                                 'exec-dir': 'execdir',
                                 'execDir': 'execdir',
                                 'algo-deterministic': 'deterministic',
                                 'paramFile': 'paramfile',
                                 'pcs-file': 'paramfile',
                                 'param-file': 'paramfile',
                                 'run-obj': 'run_obj',
                                 'run-objective': 'run_obj',
                                 'runObj': 'run_obj',
                                 'overall_obj': 'overall_obj',
                                 'intra-obj': 'overall_obj',
                                 'intra-instance-obj': 'overall_obj',
                                 'overall-obj': 'overall_obj',
                                 'intraInstanceObj': 'overall_obj',
                                 'overallObj': 'overall_obj',
                                 'intra_instance_obj': 'overall_obj',
                                 'algo-cutoff-time': 'cutoff_time',
                                 'target-run-cputime-limit': 'cutoff_time',
                                 'target_run_cputime_limit': 'cutoff_time',
                                 'cutoff-time': 'cutoff_time',
                                 'cutoffTime': 'cutoff_time',
                                 'cputime-limit': 'tunerTimeout',
                                 'cputime_limit': 'tunerTimeout',
                                 'tunertime-limit': 'tunerTimeout',
                                 'tuner-timeout': 'tunerTimeout',
                                 'tunerTimeout': 'tunerTimeout',
                                 'wallclock_limit': 'wallclock-limit',
                                 'runtime-limit': 'wallclock-limit',
                                 'runtimeLimit': 'wallclock-limit',
                                 'wallClockLimit': 'wallclock-limit',
                                 'output-dir': 'outdir',
                                 'outputDirectory': 'outdir',
                                 'instances': 'instance_file',
                                 'instance-file': 'instance_file',
                                 'instance-dir': 'instance_file',
                                 'instanceFile': 'instance_file',
                                 'i': 'instance_file',
                                 'instance_seed_file': 'instance_file',
                                 'test-instances': 'test_instance_file',
                                 'test-instance-file': 'test_instance_file',
                                 'test-instance-dir': 'test_instance_file',
                                 'testInstanceFile': 'test_instance_file',
                                 'test_instance_file': 'test_instance_file',
                                 'test_instance_seed_file': 'test_instance_file',
                                 'feature-file': 'feature_file',
                                 'instanceFeatureFile': 'feature_file',
                                 'feature_file': 'feature_file',
                                 'runcount-limit': 'runcount-limit',
                                 'runcount_limit': 'runcount-limit',
                                 'totalNumRunsLimit': 'runcount-limit',
                                 'numRunsLimit': 'runcount-limit',
                                 'numberOfRunsLimit': 'runcount-limit'
                                 }

        scenario_dict = {}
        with open(fn, 'r') as fh:
            for line in fh:
                line = line.replace("\n", "").strip(" ")
                # remove comments
                if line.find("#") > -1:
                    line = line[:line.find("#")]

                # skip empty lines
                if line == "":
                    continue
                if "=" in line:
                    tmp = line.split("=")
                    tmp = [' '.join(s.split()) for s in tmp]
                else:
                    tmp = line.split()
                scenario_dict[
                    scenario_option_names.get(tmp[0], tmp[0])] = " ".join(tmp[1:])
        return(scenario_dict)

    def read_instance_file(self, fn):
        """
            encapsulates read_instances_file of pysmac

            Parameters
            ----------
                fn: string
                     file name of instance file

            Returns
            -------
                list -- each element is a list where the first element is the
                instance name followed by additional
                information for the specific instance.
        """
        with open(fn, 'r') as fh:
            instance_names = fh.readlines()
        return([s.strip().split() for s in instance_names])

    def read_instance_features_file(self, fn):
        """
            encapsulates read_instances_file of pysmac

            Parameters
            ----------
                fn: string
                     file name of instance feature file

            Returns
            -------
                tuple -- first entry is a list of the feature names,
                second one is a dict with 'instance name' -
                'numpy array containing the features' key-value pairs
        """
        instances = {}
        with open(fn, 'r') as fh:
            lines = fh.readlines()
            for line in lines[1:]:
                tmp = line.strip().split(",")
                instances[tmp[0]] = np.array(tmp[1:], dtype=np.double)
        return(lines[0].split(",")[1:], instances)

    def read_pcs_file(self, fn):
        """
            encapsulates generating configuration space object

            Parameters
            ----------
                fn: string
                     file name of pcs file

            Returns
            -------
                Object of ConfigSpace
        """
        space = pcs.read(fn)
        return space
