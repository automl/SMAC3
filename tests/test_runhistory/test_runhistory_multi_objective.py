import os
import pickle
import tempfile
import unittest
import pytest

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.tae import StatusType
from smac.runhistory.runhistory import RunHistory

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='a',
                                                       lower=0,
                                                       upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='b',
                                                       lower=0,
                                                       upper=100))
    return cs


class RunhistoryMultiObjectiveTest(unittest.TestCase):

    def test_add_and_pickle(self):
        '''
        Simply adding some rundata to runhistory, then pickle it.
        '''
        rh = RunHistory()
        cs = get_config_space()
        config = Configuration(cs, values={'a': 1, 'b': 2})

        self.assertTrue(rh.empty())

        rh.add(config=config, cost=[10, 20], time=20,
               status=StatusType.SUCCESS, instance_id=None,
               seed=None, starttime=100, endtime=120,
               additional_info=None)

        rh.add(config=config, cost=[4.5, 5.5], time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=12354, starttime=10, endtime=30,
               additional_info={"start_time": 10})

        rh.add(config=config, cost=['4.8', '5.8'], time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=12354, starttime=10, endtime=30,
               additional_info={"start_time": 10})

        self.assertFalse(rh.empty())

        tmpfile = tempfile.NamedTemporaryFile(mode='wb', delete=False)
        pickle.dump(rh, tmpfile, -1)
        name = tmpfile.name
        tmpfile.close()

        with open(name, 'rb') as fh:
            loaded_rh = pickle.load(fh)

        self.assertEqual(loaded_rh.data, rh.data)

    def test_illegal_input(self):
        rh = RunHistory()
        cs = get_config_space()
        config = Configuration(cs, values={'a': 1, 'b': 2})

        self.assertTrue(rh.empty())

        rh.add(config=config, cost=[10, 20], time=20,
               status=StatusType.SUCCESS, instance_id=None,
               seed=None, starttime=100, endtime=120,
               additional_info=None)

        with pytest.raises(ValueError):
            rh.add(config=config, cost=[4.5, 5.5, 6.5], time=20,
                   status=StatusType.SUCCESS, instance_id=1,
                   seed=12354, starttime=10, endtime=30,
                   additional_info={"start_time": 10})

    def test_add_multiple_times(self):
        rh = RunHistory()
        cs = get_config_space()
        config = Configuration(cs, values={'a': 1, 'b': 2})

        for i in range(5):
            rh.add(config=config, cost=[i + 1, i + 2], time=i + 1,
                   status=StatusType.SUCCESS, instance_id=None,
                   seed=12345, additional_info=None)

        self.assertEqual(len(rh.data), 1)
        self.assertEqual(len(rh.get_runs_for_config(config, only_max_observed_budget=True)), 1)
        self.assertEqual(len(rh._configid_to_inst_seed_budget[1]), 1)
        self.assertEqual(list(rh.data.values())[0].cost, [1.0, 2.0])

    def test_full_update(self):
        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs,
                                values={'a': 1, 'b': 2})
        config2 = Configuration(cs,
                                values={'a': 1, 'b': 3})
        rh.add(config=config1, cost=[10, 40], time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=1)

        rh.add(config=config2, cost=[10, 40], time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=1)

        rh.add(config=config2, cost=[20, 80], time=20,
               status=StatusType.SUCCESS, instance_id=2,
               seed=2)

        cost_config2 = rh.get_cost(config2)

        rh.compute_all_costs()
        updated_cost_config2 = rh.get_cost(config2)

        for c1, c2 in zip(cost_config2, updated_cost_config2):
            self.assertEqual(c1, c2)

        rh.compute_all_costs(instances=[2])
        updated_cost_config2 = rh.get_cost(config2)

        for c1, c2 in zip(updated_cost_config2, [20, 80]):
            self.assertEqual(c1, c2)

    def test_incremental_update(self):

        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs,
                                values={'a': 1, 'b': 2})

        rh.add(config=config1, cost=[10, 100], time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=1)

        for c1, c2 in zip(rh.get_cost(config1), [10, 100]):
            self.assertEqual(c1, c2)

        rh.add(config=config1, cost=[20, 50], time=20,
               status=StatusType.SUCCESS, instance_id=2,
               seed=1)

        for c1, c2 in zip(rh.get_cost(config1), [15, 75]):
            self.assertEqual(c1, c2)

    def test_multiple_budgets(self):

        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs,
                                values={'a': 1, 'b': 2})

        rh.add(config=config1, cost=[10, 50], time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=1, budget=1)

        for c1, c2 in zip(rh.get_cost(config1), [10, 50]):
            self.assertEqual(c1, c2)

        # only the higher budget gets included in the config cost
        rh.add(config=config1, cost=[20, 25], time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=1, budget=2)

        for c1, c2 in zip(rh.get_cost(config1), [20, 25]):
            self.assertEqual(c1, c2)

        for c1, c2 in zip(rh.get_min_cost(config1), [10, 25]):
            self.assertEqual(c1, c2)

    def test_get_configs_per_budget(self):
        rh = RunHistory()
        cs = get_config_space()

        config1 = Configuration(cs,
                                values={'a': 1, 'b': 1})
        rh.add(config=config1, cost=[10, 20], time=10,
               status=StatusType.SUCCESS, instance_id=1,
               seed=1, budget=1)

        config2 = Configuration(cs,
                                values={'a': 2, 'b': 2})
        rh.add(config=config2, cost=[20, 30], time=20,
               status=StatusType.SUCCESS, instance_id=1,
               seed=1, budget=1)

        config3 = Configuration(cs,
                                values={'a': 3, 'b': 3})
        rh.add(config=config3, cost=[30, 40], time=30,
               status=StatusType.SUCCESS, instance_id=1,
               seed=1, budget=3)

        configs = rh.get_all_configs_per_budget([1])
        self.assertListEqual(configs, [config1, config2])

    def test_json_origin(self):

        for origin in ['test_origin', None]:
            rh = RunHistory()
            cs = get_config_space()
            config1 = Configuration(cs,
                                    values={'a': 1, 'b': 2},
                                    origin=origin)

            rh.add(config=config1, cost=[10., 20.], time=20,
                   status=StatusType.SUCCESS, instance_id=1,
                   seed=1)

            path = 'test/test_files/test_json_origin.json'
            rh.save_json(path)
            _ = rh.load_json(path, cs)

            self.assertEqual(rh.get_all_configs()[0].origin, origin)

            os.remove(path)


if __name__ == "__main__":
    unittest.main()
