__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"

import logging
from subprocess import Popen, PIPE


class StatusType(object):

    """
        class to define numbers for status types
    """
    SUCCESS = 1
    TIMEOUT = 2
    CRASHED = 3
    ABORT = 4


class ExecuteTARun(object):

    """
        executes a target algorithm run with a given configuration
        on a given instance and some resource limitations

        Attributes
        ----------
        ta : string
            the command line call to the target algorithm (wrapper)
    """

    def __init__(self, ta):
        """
        Constructor

        Parameters
        ----------
            ta : string
                target algorithm command line string
        """
        self.ta = ta
        self.logger = logging.getLogger("ExecuteTARun")
        pass

    def run(self, config, instance,
            cutoff=99999999999999.,
            seed=12345):
        """
            runs target algorithm <self.ta> with configuration <config> on
            instance <instance> with instance specifics <specifics>
            for at most <cutoff> seconds and random seed <seed>

            Parameters
            ----------
                config : dictionary
                    dictionary param -> value
                instance : string
                    problem instance
                cutoff : double
                    runtime cutoff
                seed : int
                    random seed
            Returns
            -------
                status: enum of StatusType (int)
                    {SUCCESS, TIMEOUT, CRASHED, ABORT}
                cost: float
                    cost/regret/quality (float) (None, if not returned by TA)
                runtime: float
                    runtime (None if not returned by TA)
        """
        return StatusType.SUCCESS, 12345.0, 1.2345
