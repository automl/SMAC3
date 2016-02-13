__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"

import os
import logging
import numpy

from smac.stats.stats import Stats


class TrajLogger(object):

    """
        writes trajectory logs files 

        Attributes
        ----------
        logger : Logger oject
    """

    def __init__(self, output_dir="smac3-output"):
        """
        Constructor 
        
        creates output directory if not exists already
        
        Arguments
        ---------
        output_dir: str
            directory for logging
        """
        self.logger = logging.getLogger("TrajLogger")

        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError:
                self.logger.error("Could not make output directory: %s" %(output_dir))
                
        self.output_dir = output_dir
        
        self.old_traj_fn = os.path.join(output_dir, "traj_old.csv")
        if not os.path.isfile(self.old_traj_fn):
            with open(self.old_traj_fn, "w") as fp:
                fp.write('"CPU Time Used","Estimated Training Performance","Wallclock Time","Incumbent ID","Automatic Configurator (CPU) Time","Configuration..."\n')
                

    def add_entry(self, train_perf, 
                  incumbent_id, incumbent):
        """
            checks command line arguments
            (e.g., whether all given files exist)

            it uses the time stats available when this function is called

            Parameters
            ----------
            train_perf: float
                estimated performance on training (sub)set 
            incumbent_id: int
                id of incumbent
            incumbent: Configuration()
                current incumbent configuration
        """
        conf = []
        for p in incumbent:  
            conf.append("%s=%s" %(p,incumbent[p]))
            
        ta_time_used = Stats.ta_time_used
        wallclock_time = Stats.get_used_wallclock_time()
        
        with open(self.old_traj_fn, "a") as fp:
            fp.write("%f, %f, %f, %d, %f, %s" %(
                                                ta_time_used,
                                                train_perf,
                                                wallclock_time,
                                                incumbent_id,
                                                wallclock_time - ta_time_used,
                                                ", ".join(conf)
                                                ))
