import os
import pickle
import tempfile
import unittest

import pytest
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.runhistory.runhistory import RunHistory
from smac.runner.abstract_runner import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name="a", lower=0, upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name="b", lower=0, upper=100))
    return cs


class RunhistoryMultiObjectiveTest(unittest.TestCase):
    def test_add_and_pickle(self):
        """
        Simply adding some rundata to runhistory, then pickle it.
        """
        rh = RunHistory()
        cs = get_config_space()
        config = Configuration(cs, values={"a": 1, "b": 2})

        self.assertTrue(rh.empty())

        rh.add(
            config=config,
            cost=[10, 20],
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
            cost=[4.5, 5.5],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=12354,
            starttime=10,
            endtime=30,
            additional_info={"start_time": 10},
        )

        rh.add(
            config=config,
            cost=["4.8", "5.8"],
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
        cs = get_config_space()
        config = Configuration(cs, values={"a": 1, "b": 2})

        self.assertTrue(rh.empty())

        with pytest.raises(ValueError):
            rh.add(
                config=config,
                cost=[4.5, 5.5, 6.5],
                time=20,
                status=StatusType.SUCCESS,
                instance_id=1,
                seed=12354,
                starttime=10,
                endtime=30,
                additional_info={"start_time": 10},
            )

            rh.add(
                config=config,
                cost=[2.5, 5.5],
                time=20,
                status=StatusType.SUCCESS,
                instance_id=1,
                seed=12354,
                starttime=10,
                endtime=30,
                additional_info={"start_time": 10},
            )

    def test_add_multiple_times(self):
        rh = RunHistory()
        cs = get_config_space()
        config = Configuration(cs, values={"a": 1, "b": 2})

        for i in range(5):
            rh.add(
                config=config,
                cost=[i + 1, i + 2],
                time=i + 1,
                status=StatusType.SUCCESS,
                instance_id=None,
                seed=12345,
                additional_info=None,
            )

        self.assertEqual(len(rh.data), 1)
        self.assertEqual(len(rh.get_runs_for_config(config, only_max_observed_budget=True)), 1)
        self.assertEqual(len(rh._configid_to_inst_seed_budget[1]), 1)

        # We expect to get 1.0 and 2.0 because runhistory does not overwrite by default
        self.assertEqual(list(rh.data.values())[0].cost, [1.0, 2.0])

    def test_full(self):
        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})
        config2 = Configuration(cs, values={"a": 1, "b": 3})
        config3 = Configuration(cs, values={"a": 1, "b": 4})
        rh.add(
            config=config1,
            cost=[50, 100],
            time=20,
            status=StatusType.SUCCESS,
        )

        # Only one value: Normalization goes to 1.0
        self.assertEqual(rh.get_cost(config1), 1.0)

        rh.add(
            config=config2,
            cost=[150, 50],
            time=30,
            status=StatusType.SUCCESS,
        )

        # The cost of the first config must be updated
        # We would expect [0, 1] and the normalized value would be 0.5
        self.assertEqual(rh.get_cost(config1), 0.5)

        # We would expect [1, 0] and the normalized value would be 0.5
        self.assertEqual(rh.get_cost(config2), 0.5)

        rh.add(
            config=config3,
            cost=[100, 0],
            time=40,
            status=StatusType.SUCCESS,
        )

        # [0, 1] -> 0.5
        self.assertEqual(rh.get_cost(config1), 0.5)

        # [1, 0.5] -> 0.75
        self.assertEqual(rh.get_cost(config2), 0.75)

        # [0.5, 0] -> 0.25
        self.assertEqual(rh.get_cost(config3), 0.25)

    def test_full_update(self):
        rh = RunHistory(overwrite_existing_runs=True)
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})
        config2 = Configuration(cs, values={"a": 1, "b": 3})
        rh.add(
            config=config1,
            cost=[10, 40],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
        )

        rh.add(
            config=config1,
            cost=[0, 100],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=2,
            seed=2,
        )

        rh.add(
            config=config2,
            cost=[10, 40],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
        )

        rh.add(
            config=config2,
            cost=[20, 80],
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

        self.assertAlmostEqual(updated_cost_config2, 0.833, places=3)

    def test_incremental_update(self):

        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})

        rh.add(
            config=config1,
            cost=[10, 100],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
        )

        self.assertEqual(rh.get_cost(config1), 1.0)

        rh.add(
            config=config1,
            cost=[20, 50],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=2,
            seed=1,
        )

        # We don't except moving average of 0.75 here because
        # the costs should always be updated.
        self.assertEqual(rh.get_cost(config1), 0.5)

        rh.add(
            config=config1,
            cost=[0, 100],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=3,
            seed=1,
        )

        self.assertAlmostEqual(rh.get_cost(config1), 0.583, places=3)

    def test_multiple_budgets(self):

        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})

        rh.add(
            config=config1,
            cost=[10, 50],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        self.assertEqual(rh.get_cost(config1), 1.0)

        # Only the higher budget gets included in the config cost
        # However, we expect that the bounds are changed
        rh.add(
            config=config1,
            cost=[20, 25],
            time=25,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=5,
        )

        self.assertEqual(rh.get_cost(config1), 0.5)

    def test_get_configs_per_budget(self):
        rh = RunHistory()
        cs = get_config_space()

        config1 = Configuration(cs, values={"a": 1, "b": 1})
        rh.add(
            config=config1,
            cost=[10, 20],
            time=10,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        config2 = Configuration(cs, values={"a": 2, "b": 2})
        rh.add(
            config=config2,
            cost=[20, 30],
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        config3 = Configuration(cs, values={"a": 3, "b": 3})
        rh.add(
            config=config3,
            cost=[30, 40],
            time=30,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=3,
        )

        configs = rh.get_all_configs_per_budget([1])
        self.assertListEqual(configs, [config1, config2])

    def test_json_origin(self):
        for origin in ["test_origin", None]:
            rh = RunHistory()
            cs = get_config_space()
            config1 = Configuration(cs, values={"a": 1, "b": 2}, origin=origin)

            rh.add(
                config=config1,
                cost=[10.0, 20.0],
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

    def test_objective_bounds(self):
        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})
        config2 = Configuration(cs, values={"a": 2, "b": 3})
        config3 = Configuration(cs, values={"a": 3, "b": 4})

        rh.add(
            config=config1,
            cost=[10, 50],
            time=5,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        rh.add(
            config=config2,
            cost=[5, 100],
            time=10,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        rh.add(
            config=config3,
            cost=[7.5, 150],
            time=15,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        self.assertEqual(rh.objective_bounds[0], (5, 10))
        self.assertEqual(rh.objective_bounds[1], (50, 150))

        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})
        config2 = Configuration(cs, values={"a": 2, "b": 3})
        config3 = Configuration(cs, values={"a": 3, "b": 4})

        rh.add(
            config=config1,
            cost=10,
            time=5,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
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
            config=config3,
            cost=7.5,
            time=15,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        self.assertEqual(rh.objective_bounds[0], (5, 10))

    def test_bounds_on_crash(self):
        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})
        config2 = Configuration(cs, values={"a": 2, "b": 3})
        config3 = Configuration(cs, values={"a": 3, "b": 4})

        rh.add(
            config=config1,
            cost=[10, 50],
            time=5,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        rh.add(
            config=config2,
            cost=[100, 100],
            time=10,
            status=StatusType.CRASHED,
            instance_id=1,
            seed=1,
            budget=1,
        )

        rh.add(
            config=config3,
            cost=[0, 150],
            time=15,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=1,
        )

        self.assertEqual(rh.objective_bounds[0], (0, 10))
        self.assertEqual(rh.objective_bounds[1], (50, 150))

    def test_instances(self):
        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})
        config2 = Configuration(cs, values={"a": 2, "b": 3})

        rh.add(
            config=config1,
            cost=[0, 10],
            time=5,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=0,
        )

        rh.add(
            config=config1,
            cost=[50, 20],
            time=10,
            status=StatusType.SUCCESS,
            instance_id=2,
            seed=1,
            budget=0,
        )

        rh.add(
            config=config1,
            cost=[75, 20],
            time=10,
            status=StatusType.SUCCESS,
            instance_id=3,
            seed=1,
            budget=0,
        )

        rh.add(
            config=config2,
            cost=[100, 30],
            time=15,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=0,
        )

        rh.add(
            config=config2,
            cost=[0, 30],
            time=15,
            status=StatusType.SUCCESS,
            instance_id=2,
            seed=1,
            budget=0,
        )

        self.assertEqual(rh.objective_bounds[0], (0, 100))
        self.assertEqual(rh.objective_bounds[1], (10, 30))

        # Average cost returns us the cost of the latest budget
        self.assertEqual(rh.get_cost(config1), 0.375)
        self.assertEqual(rh.get_cost(config2), 0.75)

    def test_budgets(self):
        rh = RunHistory()
        cs = get_config_space()
        config1 = Configuration(cs, values={"a": 1, "b": 2})
        config2 = Configuration(cs, values={"a": 2, "b": 3})

        rh.add(
            config=config1,
            cost=[0, 50],
            time=5,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=5,
        )

        rh.add(
            config=config1,
            cost=[40, 100],
            time=10,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=15,
        )

        # SMAC does not overwrite by default
        rh.add(
            config=config1,
            cost=[502342352, 23425234],
            time=11,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=15,
        )

        rh.add(
            config=config2,
            cost=[0, 150],
            time=15,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=1,
            budget=5,
        )

        self.assertEqual(rh.objective_bounds[0], (0, 40))
        self.assertEqual(rh.objective_bounds[1], (50, 150))

        # Average cost returns us the cost of the latest budget
        self.assertEqual(rh.get_cost(config1), 0.75)
        self.assertEqual(rh.average_cost(config1), [40.0, 100.0])

        self.assertEqual(rh.get_cost(config2), 0.5)
        self.assertEqual(rh.average_cost(config2), [0, 150])


if __name__ == "__main__":
    t = RunhistoryMultiObjectiveTest()
    t.test_budgets()
