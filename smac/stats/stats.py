import time
import logging

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class Stats(object):

    '''
        all statistics collected during configuration run
    '''
    
    def __init__(self, scenario):
          
        self.__scenario = scenario
        
        self.ta_runs = 0
        self.wallclock_time_used = 0
        self.ta_time_used = 0
        self.inc_changed = 0
    
        self._start_time = None
        self._logger = logging.getLogger("Stats")

    def start_timing(self):
        '''
            starts the timer (for the runtime configuration budget)
        '''
        if self.__scenario:
            self._start_time = time.time()
        else:
            raise ValueError("Scenario is missing")

    def get_used_wallclock_time(self):
        '''
            returns used wallclock time
            Returns
            -------
            wallclock_time : int
                used wallclock time in sec
        '''
        
        return time.time() - self._start_time

    def get_remaing_time_budget(self):
        '''
            subtracts the runtime configuration budget with the used wallclock time
        '''
        if self.__scenario:
            return self.__scenario.wallclock_limit - (time.time() - self._start_time)
        else:
            raise "Scenario is missing"

    def get_remaining_ta_runs(self):
        '''
           subtract the target algorithm runs in the scenario with the used ta runs 
        '''
        if self.__scenario:
            return self.__scenario.ta_run_limit - self.ta_runs
        else:
            raise "Scenario is missing"

    def get_remaining_ta_budget(self):
        '''
            subtracts the ta running budget with the used time
        '''
        if self.__scenario:
            return self.__scenario.algo_runs_timelimit - self.ta_time_used
        
    def is_budget_exhausted(self):
        '''
            check whether the configuration budget for time budget, ta_budget and ta_runs is empty 
            
            Returns
            -------
                true if one of the budgets is exhausted
        '''
        return  self.get_remaing_time_budget() < 0 or \
                self.get_remaining_ta_budget() < 0 or \
                self.get_remaining_ta_runs() <= 0

    def print_stats(self, debug_out:bool=False):
        '''
            prints all statistics
            
            Arguments
            ---------
            debug: bool
                use logging.debug instead of logging.info if set to true
        '''
        log_func = self._logger.info
        if debug_out:
            log_func = self._logger.debug
        
        log_func("##########################################################")
        log_func("Statistics:")
        log_func("#Incumbent changed: %d" %(self.inc_changed - 1)) # first change is default conf
        log_func("#Target algorithm runs: %d / %s" %(self.ta_runs, str(self.__scenario.ta_run_limit)))
        log_func("Used wallclock time: %.2f / %.2f sec " %(time.time() - self._start_time, self.__scenario.wallclock_limit))
        log_func("Used target algorithm runtime: %.2f / %.2f sec" %(self.ta_time_used, self.__scenario.algo_runs_timelimit))
        
        log_func("##########################################################")    
