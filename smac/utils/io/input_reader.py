import numpy as np
from smac.configspace import pcs

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class InputReader(object):

    """Reading all input files for SMAC (scenario file, instance files, ...)

    Note: Most of this code was taken from the pysmac repository.
          We copy it here because we don't want smac3 to depend
          on an earlier version!
    """

    def __init__(self):
        pass

    def read_scenario_file(self, fn: str):
        """Encapsulates read_scenario_file of pysmac

        Parameters
        ----------
            fn: string
                 File name of scenario file

        Returns
        -------
        dict : dictionary
            (key, value) pairs are (variable name, variable value)
        """
        # translate the difference option names to a canonical name
        # kept for backwards-compatibility
        scenario_option_names = {'algo-exec': 'algo',
                                 'algoExec': 'algo',
                                 'algo': 'algo',
                                 'algo-exec-dir': 'execdir',
                                 'exec-dir': 'execdir',
                                 'execDir': 'execdir',
                                 'execdir': 'execdir',
                                 'algo-deterministic': 'deterministic',
                                 'deterministic': 'deterministic',
                                 'paramFile': 'paramfile',
                                 'pcs-file': 'paramfile',
                                 'param-file': 'paramfile',
                                 'paramfile': 'paramfile',
                                 'run-obj': 'run_obj',
                                 'run-objective': 'run_obj',
                                 'runObj': 'run_obj',
                                 'run_obj': 'run_obj',
                                 'overall_obj': 'overall_obj',
                                 'intra-obj': 'overall_obj',
                                 'intra-instance-obj': 'overall_obj',
                                 'overall-obj': 'overall_obj',
                                 'intraInstanceObj': 'overall_obj',
                                 'overallObj': 'overall_obj',
                                 'intra_instance_obj': 'overall_obj',
                                 'cost-for-crash': 'cost_for_crash',
                                 'cost_for_crash': 'cost_for_crash',
                                 'algo-cutoff-time': 'cutoff_time',
                                 'target-run-cputime-limit': 'cutoff_time',
                                 'target_run_cputime_limit': 'cutoff_time',
                                 'cutoff-time': 'cutoff_time',
                                 'cutoffTime': 'cutoff_time',
                                 'cutoff_time': 'cutoff_time',
                                 'memory-limit': 'memory_limit',
                                 'memory_limit': 'memory_limit',
                                 'cputime-limit': 'tuner_timeout',
                                 'cputime_limit': 'tuner_timeout',
                                 'tunertime-limit': 'tuner_timeout',
                                 'tuner-timeout': 'tuner_timeout',
                                 'tunerTimeout': 'tuner_timeout',
                                 'tuner_timeout': 'tuner_timeout',
                                 'wallclock-limit': 'wallclock_limit',
                                 'runtime-limit': 'wallclock_limit',
                                 'runtimeLimit': 'wallclock_limit',
                                 'wallClockLimit': 'wallclock_limit',
                                 'wallclock_limit': 'wallclock_limit',
                                 'output-dir': 'output_dir',
                                 'outputDirectory': 'output_dir',
                                 'outdir': 'output_dir',
                                 'output_dir': 'output_dir',
                                 'instances': 'instance_file',
                                 'instance-file': 'instance_file',
                                 'instance-dir': 'instance_file',
                                 'instanceFile': 'instance_file',
                                 'instance_file': 'instance_file',
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
                                 'runcount-limit': 'runcount_limit',
                                 'runcount_limit': 'runcount_limit',
                                 'totalNumRunsLimit': 'runcount_limit',
                                 'numRunsLimit': 'runcount_limit',
                                 'numberOfRunsLimit': 'runcount_limit',
                                 'initial-incumbent': 'initial_incumbent',
                                 'initial_incumbent': 'initial_incumbent'
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

    def read_instance_file(self, fn: str):
        """Encapsulates read_instances_file of pysmac

        Parameters
        ----------
            fn: string
                 File name of instance file

        Returns
        -------
            instances: list
                Each element is a list where the first element is the
                instance name followed by additional
                information for the specific instance.
        """
        with open(fn, 'r') as fh:
            instance_names = fh.readlines()
        return([s.strip().split() for s in instance_names])

    def read_instance_features_file(self, fn: str):
        """Encapsulates read_instances_file of pysmac

        Parameters
        ----------
            fn: string
                File name of instance feature file

        Returns
        -------
            features: tuple
                first entry is a list of the feature names,
                second one is a dict with 'instance name' -
                'numpy array containing the features' key-value pairs
        """
        instances = {}
        with open(fn, 'r') as fh:
            lines = fh.readlines()
            for line in lines[1:]:
                tmp = line.strip().split(",")
                instances[tmp[0]] = np.array(tmp[1:], dtype=np.double)
        return [f.strip() for f in lines[0].rstrip("\n").split(",")[1:]], instances

    def read_pcs_file(self, fn: str):
        """Encapsulates generating configuration space object

        Parameters
        ----------
            fn: string
                 File name of pcs file

        Returns
        -------
            ConfigSpace: ConfigSpace
        """
        space = pcs.read(fn)
        return space
