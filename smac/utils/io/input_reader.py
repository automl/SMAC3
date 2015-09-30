__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"

from pysmac.utils.smac_input_readers import read_scenario_file as rsf
from pysmac.utils.smac_output_readers import read_instances_file as rif
from pysmac.utils.smac_output_readers import read_instance_features_file as rff


class InputReader(object):

    """
        reading all input files for SMAC (scenario file, instance files, ...)

        TODO: I don't know whether this class is necessary at all
        since pysmac implements it already
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
        return rsf(fn)

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
        return rif(fn)

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
        return rff(fn)

    def read_pcs_file(self, fn):
        """
            encapsulates generating configuration space object

            Parameters
            ----------
                fn: string
                     file name of pcs file

            TODO: add interface to Matthias' ConfigSpace

            Returns
            -------
                Object of ConfigSpace
        """
        # cs = ConfigSpace()
        # cs.read_file(fn)
        return None
