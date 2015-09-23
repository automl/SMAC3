'''
Created on Sep 23, 2015

@author: lindauer
'''

from subprocess import Popen, PIPE


class ExecuteTARun(object):
    '''
        executes a target algorithm run with a given configuration
        on a given instance and some resource limitations
    '''

    def __init__(self, ta):
        '''
        Constructor
        Args:
            ta : target algorithm (string)
        '''
        self.ta = ta
        pass

    def run(self, config, instance, specifics="0",
            cutoff=99999999999999.,
            seed=12345):
        '''
            runs target algorithm <self.ta> with configuration <config> on
            instance <instance> with instance specifics <specifics>
            for at most <cutoff> seconds and random seed <seed>
            Args:
                config : dictionary param -> value
                instance: problem instance (string)
                specifics: instance specifics (string)
                cutoff: runtime cutoff (double)
                seed : random seed (integer)
            Return:
                status: one of SUCCESS, TIMEOUT, CRASHED, ABORT (string)
                cost: cost/regret/quality (float) (None, if not returned by TA)
                runtime: runtime (float; None if not returned by TA)
        '''
        return "SUCCESS", 12345, 1
