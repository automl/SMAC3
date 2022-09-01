from typing import Optional

import os
import pickle
import tempfile
import unittest

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.runhistory.runhistory import RunHistory, TrialKey
from smac.runner.abstract_runner import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name="a", lower=0, upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name="b", lower=0, upper=100))
    return cs


class RunhistoryTest(unittest.TestCase):
    def test_add_and_pickle(self):
        """
        simply adding some rundata to runhistory, then pickle it
        """
        rh = RunHistory()
        cs = get_config_space()
        config = Configuration(cs, values={"a": 1, "b": 2})

        self.assertTrue(rh.empty())

        rh.add(
            config=config,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=None,
            seed=None,
            starttime=100,
            endtime=120,
            additional_info=None,
        )

        rh.add(
            config=config,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=12354,
            starttime=10,
            endtime=30,
            additional_info={"start_time": 10},
        )

        self.assertFalse(rh.empty())

        tmpfile = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        pickle.dump(rh, tmpfile, -1)
        name = tmpfile.name
        tmpfile.close()

        with open(name, "rb") as fh:
            loaded_rh = pickle.load(fh)  # nosec
        self.assertEqual(loaded_rh.data, rh.data)

    def test_illegal_input(self):
        rh = RunHistory()

        with self.assertRaisesRegex(TypeError, "Configuration to add to the runhistory must not be None"):
            rh.add(config=None, cost=1.23, time=2.34, status=StatusType.SUCCESS)

        with self.assertRaisesRegex(
            TypeError,
            "Configuration to add to the runhistory is not of type Configuration, but <class 'str'>",
        ):
            rh.add(config="abc", cost=1.23, time=2.34, status=StatusType.SUCCESS)

    def test_add_multiple_times(self):
        rh = RunHistory()
        cs = get_config_space()
        config = Configuration(cs, values={"a": 1, "b": 2})

        for i in range(5):
            rh.add(
                config=config,
                cost=i + 1,
                time=i + 1,
                status=StatusType.SUCCESS,
                instance_id=None,
                seed=12345,
                additional_info=None,
                budget=0,
            )

        self.assertEqual(len(rh.data), 1)
        self.assertEqual(len(rh.get_runs_for_config(config, only_max_observed_budget=True)), 1)
        self.assertEqual(len(rh._configid_to_inst_seed_budget[1]), 1)
        self.assertEqual(list(rh.data.values())[0].cost, 1)

    def test_get_config_runs(self):
        """
        get some config runs from runhistory
        """
        # return max observed budget only
        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})
        config2 = Configuration(cs, values={"a": 1, "b": 3})
        rh.add(
            config=config1,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )
        rh.add(
            config=config1,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=2,
        )
        with self.assertRaisesRegex(ValueError, "This should not happen!"):
            rh.add(
                config=config1,
                cost=10,
                time=20,
                status=StatusType.SUCCESS,
                instance_id=2,
                seed=2,
                budget=1,
            )

        rh.add(
            config=config2,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        ist = rh.get_runs_for_config(config=config1, only_max_observed_budget=True)

        self.assertEqual(len(ist), 2)
        self.assertEqual(ist[0].instance, 1)
        self.assertEqual(ist[1].instance, 2)
        self.assertEqual(ist[0].budget, 2)
        self.assertEqual(ist[1].budget, 1)

        # multiple budgets (only_max_observed_budget=False)
        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})
        config2 = Configuration(cs, values={"a": 1, "b": 3})
        rh.add(
            config=config1,
            cost=5,
            time=10,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )
        rh.add(
            config=config1,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=2,
        )

        rh.add(
            config=config2,
            cost=5,
            time=10,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )
        rh.add(
            config=config2,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=2,
        )

        ist = rh.get_runs_for_config(config=config1, only_max_observed_budget=False)

        self.assertEqual(len(ist), 2)
        self.assertEqual(ist[0].instance, 1)
        self.assertEqual(ist[0].budget, 1)
        self.assertEqual(ist[1].budget, 2)

    def test_full_update(self):
        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})
        config2 = Configuration(cs, values={"a": 1, "b": 3})
        rh.add(
            config=config1,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
        )

        rh.add(
            config=config2,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
        )

        rh.add(
            config=config2,
            cost=20,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=2,
            seed=2,
        )

        cost_config2 = rh.get_cost(config2)

        rh.compute_all_costs()
        updated_cost_config2 = rh.get_cost(config2)
        self.assertEqual(cost_config2, updated_cost_config2)

        rh.compute_all_costs(instances=[2])
        updated_cost_config2 = rh.get_cost(config2)
        self.assertNotEqual(cost_config2, updated_cost_config2)
        self.assertEqual(updated_cost_config2, 20)

        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})
        config2 = Configuration(cs, values={"a": 1, "b": 3})
        rh.add(
            config=config1,
            cost=[10],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
        )

        rh.add(
            config=config2,
            cost=[10],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
        )

        rh.add(
            config=config2,
            cost=[20],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=2,
            seed=2,
        )

        cost_config2 = rh.get_cost(config2)

        rh.compute_all_costs()
        updated_cost_config2 = rh.get_cost(config2)
        self.assertEqual(cost_config2, updated_cost_config2)

        rh.compute_all_costs(instances=[2])
        updated_cost_config2 = rh.get_cost(config2)
        self.assertNotEqual(cost_config2, updated_cost_config2)
        self.assertEqual(updated_cost_config2, 20)

    def test_incremental_update(self):

        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})

        rh.add(
            config=config1,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
        )

        self.assertEqual(rh.get_cost(config1), 10)

        rh.add(
            config=config1,
            cost=20,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=2,
            seed=1,
        )

        self.assertEqual(rh.get_cost(config1), 15)

    def test_multiple_budgets(self):

        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})

        rh.add(
            config=config1,
            cost=10,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        self.assertEqual(rh.get_cost(config1), 10)

        # only the higher budget gets included in the config cost
        rh.add(
            config=config1,
            cost=20,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=2,
        )

        self.assertEqual(rh.get_cost(config1), 20)
        self.assertEqual(rh.get_min_cost(config1), 10)

    def test_get_configs_per_budget(self):

        rh = RunHistory()
        cs = get_config_space()

        config1 = Configuration(cs, values={"a": 1, "b": 1})
        rh.add(
            config=config1,
            cost=10,
            time=10,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        config2 = Configuration(cs, values={"a": 2, "b": 2})
        rh.add(
            config=config2,
            cost=20,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        config3 = Configuration(cs, values={"a": 3, "b": 3})
        rh.add(
            config=config3,
            cost=30,
            time=30,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=3,
        )

        self.assertListEqual(rh.get_all_configs_per_budget([1]), [config1, config2])

    def test_json_origin(self):

        for origin in ["test_origin", None]:
            rh = RunHistory()
            cs = get_config_space()
            config1 = Configuration(cs, values={"a": 1, "b": 2}, origin=origin)

            rh.add(
                config=config1,
                cost=10,
                time=20,
                status=StatusType.SUCCESS,
                instance_id=1,
                seed=1,
            )

            path = "tests/test_files/test_json_origin.json"
            rh.save_json(path)
            _ = rh.load_json(path, cs)

            self.assertEqual(rh.get_all_configs()[0].origin, origin)

            os.remove(path)


class RunHistoryMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cs = get_config_space()
        self.runhistory = RunHistory()

    def add_item(
        self,
        cost: float = 0.0,
        time: float = 1.0,
        seed: int = 0,
        instance_id: Optional[str] = None,
        budget: float = 0.0,
        status=StatusType.SUCCESS,
    ) -> TrialKey:
        """No easy way to generate a key before hand"""
        self.runhistory.add(
            config=self.cs.sample_configuration(),
            cost=cost,
            time=time,
            instance_id=instance_id,
            seed=seed,
            budget=budget,
            status=status,
        )
        return TrialKey(
            config_id=self.runhistory._n_id,  # What's used internally during `add`
            instance_id=instance_id,
            seed=seed,
            budget=budget,
        )

    def test_contains(self):
        """Test that keys added are contained `in` and not if not added"""
        k = self.add_item()
        assert k in self.runhistory

        new_rh = RunHistory()
        assert k not in new_rh

    def test_getting(self):
        """Test that rh[k] will return the correct value"""
        k = self.add_item(cost=1.0)
        k2 = self.add_item(cost=2.0)

        v = self.runhistory[k]
        assert v.cost == 1.0

        v2 = self.runhistory[k2]
        assert v2.cost == 2.0

    def test_len(self):
        """Test that adding items will increase the length monotonically"""
        assert len(self.runhistory) == 0

        n_items = 5

        for i in range(n_items):
            assert len(self.runhistory) == i

            self.add_item()

            assert len(self.runhistory) == i + 1

        assert len(self.runhistory) == n_items

    def test_iter(self):
        """Test that iter goes in the order of insertion and has consitent length
        with the runhistory's advertised length and it's internal `data`
        """
        params = [
            {"instance_id": "a", "cost": 1.0},
            {"instance_id": "b", "cost": 2.0},
            {"instance_id": "c", "cost": 3.0},
            {"instance_id": "d", "cost": 4.0},
        ]

        for p in params:
            self.add_item(**p)

        expected_id_order = [p["instance_id"] for p in params]
        assert [k.instance_id for k in iter(self.runhistory)] == expected_id_order

        assert len(list(iter(self.runhistory))) == len(self.runhistory)
        assert len(list(iter(self.runhistory))) == len(self.runhistory.data)

    def test_items(self):
        """Test that items goes in correct insertion order and returns key values
        as expected.
        """
        params = [
            {"instance_id": "a", "cost": 1.0},
            {"instance_id": "b", "cost": 2.0},
            {"instance_id": "c", "cost": 3.0},
            {"instance_id": "d", "cost": 4.0},
        ]

        for p in params:
            self.add_item(**p)

        for (k, v), expected in zip(self.runhistory.items(), params):
            assert k.instance_id == expected["instance_id"]
            assert v.cost == expected["cost"]

    def test_unpack(self):
        """Test that unpacking maintains order and returns key values as expected"""
        params = [
            {"instance_id": "a", "cost": 1.0},
            {"instance_id": "b", "cost": 2.0},
            {"instance_id": "c", "cost": 3.0},
            {"instance_id": "d", "cost": 4.0},
        ]

        for p in params:
            self.add_item(**p)

        unpacked = {**self.runhistory}

        for (k, v), expected in zip(unpacked.items(), params):
            assert k.instance_id == expected["instance_id"]
            assert v.cost == expected["cost"]


if __name__ == "__main__":
    unittest.main()
