import unittest
import logging

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
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
        rh = RunHistory(aggregate_func=average_cost)
        cs = get_config_space()
        config = Configuration(cs,
                               values={'a': 1, 'b': 2})

        self.assertTrue(rh.empty())

        rh.add(config=config, cost=10, time=20,
               status=StatusType.SUCCESS, instance_id=None,
               seed=None,
               additional_info=None)

        rh.add(config=config, cost=10, time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=12354,
               additional_info={"start_time": 10})

        self.assertFalse(rh.empty())

    def test_get_config_runs(self):
        '''
            get some config runs from runhistory
        '''

        rh = RunHistory(aggregate_func=average_cost)
        cs = get_config_space()
        config1 = Configuration(cs,
                                values={'a': 1, 'b': 2})
        config2 = Configuration(cs,
                                values={'a': 1, 'b': 3})
        rh.add(config=config1, cost=10, time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=1)

        rh.add(config=config2, cost=10, time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=1)

        rh.add(config=config1, cost=10, time=20,
               status=StatusType.SUCCESS, instance_id=2,
               seed=2)

        ist = rh.get_runs_for_config(config=config1)
        #print(ist)
        #print(ist[0])
        #print(ist[1])
        self.assertEqual(len(ist), 2)
        self.assertEqual(ist[0].instance, 1)
        self.assertEqual(ist[1].instance, 2)


if __name__ == "__main__":
    unittest.main()
