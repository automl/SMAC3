import os
import pickle
import tempfile
import unittest
import numpy

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost


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

    def test_add_and_pickle(self):
        '''
            simply adding some rundata to runhistory, then pickle it
        '''
        rh = RunHistory(aggregate_func=average_cost)
        cs = get_config_space()
        config = Configuration(cs, values={'a': 1, 'b': 2})

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

        tmpfile = tempfile.NamedTemporaryFile(mode='wb', delete=False)
        pickle.dump(rh, tmpfile, -1)
        name = tmpfile.name
        tmpfile.close()

        with open(name, 'rb') as fh:
            loaded_rh = pickle.load(fh)
        self.assertEqual(loaded_rh.data, rh.data)

    def test_add_multiple_times(self):
        rh = RunHistory(aggregate_func=average_cost)
        cs = get_config_space()
        config = Configuration(cs, values={'a': 1, 'b': 2})

        for i in range(5):
            rh.add(config=config, cost=i + 1, time=i + 1,
                   status=StatusType.SUCCESS, instance_id=None,
                   seed=12345, additional_info=None)

        self.assertEqual(len(rh.data), 1)
        self.assertEqual(len(rh.get_runs_for_config(config)), 1)
        self.assertEqual(len(rh._configid_to_inst_seed[1]), 1)
        self.assertEqual(list(rh.data.values())[0].cost, 1)

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

    def test_full_update(self):
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

        rh.add(config=config2, cost=20, time=20,
               status=StatusType.SUCCESS, instance_id=2,
               seed=2)

        cost_config2 = rh.get_cost(config2)

        rh.compute_all_costs()
        updated_cost_config2 = rh.get_cost(config2)
        self.assertTrue(cost_config2 == updated_cost_config2)

        rh.compute_all_costs(instances = [2])
        updated_cost_config2 = rh.get_cost(config2)
        self.assertTrue(cost_config2 != updated_cost_config2)
        self.assertTrue(updated_cost_config2==20)

    def test_incremental_update(self):

        rh = RunHistory(aggregate_func=average_cost)
        cs = get_config_space()
        config1 = Configuration(cs,
                                values={'a': 1, 'b': 2})

        rh.add(config=config1, cost=10, time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=1)

        self.assertTrue(rh.get_cost(config1) == 10)

        rh.add(config=config1, cost=20, time=20,
               status=StatusType.SUCCESS, instance_id=2,
               seed=1)

        self.assertTrue(rh.get_cost(config1) == 15)

    def test_json_origin(self):

        for origin in ['test_origin', None]:
            rh = RunHistory(aggregate_func=average_cost)
            cs = get_config_space()
            config1 = Configuration(cs,
                                    values={'a': 1, 'b': 2},
                                    origin=origin)

            rh.add(config=config1, cost=10, time=20,
                   status=StatusType.SUCCESS, instance_id=1,
                   seed=1)

            path = 'test/test_files/test_json_origin.json'
            rh.save_json(path)
            new_rh = rh.load_json(path, cs)

            self.assertEqual(rh.get_all_configs()[0].origin, origin)

            os.remove(path)


if __name__ == "__main__":
    unittest.main()
