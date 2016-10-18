__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "GPLv3"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"

import unittest
from smac.tae.execute_ta_run import StatusType
from smac.runhistory import runhistory, runhistory2epm

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.smbo.objective import average_cost

def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='a',
                                                       lower=0,
                                                       upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='b',
                                                       lower=0,
                                                       upper=100))
    return cs

class RunhistoryTest(unittest.TestCase):

    def test_add(self):
        '''
            simply adding some rundata to runhistory
        '''
        rh = runhistory.RunHistory(aggregate_func=average_cost)
        cs = get_config_space()
        config1 = Configuration(cs,
                                values={'a': 1, 'b': 2})
        config2 = Configuration(cs,
                                values={'a': 1, 'b': 25})
        config3 = Configuration(cs,
                                values={'a': 2, 'b': 2})
        rh.add(config=config1, cost=10, time=20,
               status=StatusType.SUCCESS, instance_id=23,
               seed=None,
               additional_info=None)
        rh.add(config=config2, cost=10, time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=12354,
               additional_info={"start_time": 10})
        rh.add(config=config3, cost=10, time=20,
               status=StatusType.TIMEOUT, instance_id=1,
               seed=45,
               additional_info={"start_time": 10})
        
        scen = Scenario({"cutoff_time": 10, 'cs': cs})

        self.assertRaises(TypeError, runhistory2epm.RunHistory2EPM4LogCost)

        rh2epm = runhistory2epm.RunHistory2EPM4LogCost(num_params=2,
                                                       scenario=scen)
        rhArr = rh2epm.transform(rh)

        # We expect
        #  1  2 23     0 0 23
        #  1 25  1  -> 0 1  1
        # 21  2  1     1 0  1
        #TODO: ML I don't understand this comment

if __name__ == "__main__":
    unittest.main()
