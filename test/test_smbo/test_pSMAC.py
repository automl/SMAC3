import json
import os
import shutil
import unittest
import logging

from smac.runhistory.runhistory import RunHistory, RunKey, DataOrigin
from smac.utils import test_helpers
from smac.tae.execute_ta_run import StatusType
from smac.optimizer import pSMAC
from smac.optimizer.objective import average_cost


class TestPSMAC(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), '.test_pSMAC')
        self._remove_tmp_dir()
        os.makedirs(self.tmp_dir)

    def tearDown(self):
        self._remove_tmp_dir()

    def _remove_tmp_dir(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except:
            pass

    def test_write(self):
        # The nulls make sure that we correctly emit the python None value
        fixture = '{"data": [[[1, "branin", 1], [1, 1, {"__enum__": ' \
                  '"StatusType.SUCCESS"}, null]], ' \
                  '[[1, "branini", 1], [1, 1, {"__enum__": ' \
                  '"StatusType.SUCCESS"}, null]], ' \
                  '[[2, "branini", 1], [1, 1, {"__enum__": ' \
                  '"StatusType.SUCCESS"}, null]], ' \
                  '[[2, null, 1], [1, 1, {"__enum__": ' \
                  '"StatusType.SUCCESS"}, null]], ' \
                  '[[3, "branin-hoo", 1], [1, 1, {"__enum__": ' \
                  '"StatusType.SUCCESS"}, null]], ' \
                  '[[4, null, 1], [1, 1, {"__enum__": ' \
                  '"StatusType.SUCCESS"}, null]]],' \
                  '"config_origins": {},' \
                  '"configs": {' \
                  '"4": {"x": -2.2060968293349363, "y": 5.183410905645716}, ' \
                  '"3": {"x": -2.7986616377433045, "y": 1.385078921531967}, ' \
                  '"1": {"x": 1.2553300705386103, "y": 10.804867401632372}, ' \
                  '"2": {"x": -4.998284377739827, "y": 4.534988589477597}}}'

        run_history = RunHistory(aggregate_func=average_cost)
        configuration_space = test_helpers.get_branin_config_space()
        configuration_space.seed(1)

        config = configuration_space.sample_configuration()
        # Config on two instances
        run_history.add(config, 1, 1, StatusType.SUCCESS, seed=1,
                        instance_id='branin')
        run_history.add(config, 1, 1, StatusType.SUCCESS, seed=1,
                        instance_id='branini')
        config_2 = configuration_space.sample_configuration()
        # Another config on a known instance
        run_history.add(config_2, 1, 1, StatusType.SUCCESS, seed=1,
                        instance_id='branini')
        # Known Config on no instance
        run_history.add(config_2, 1, 1, StatusType.SUCCESS, seed=1)
        # New config on new instance
        config_3 = configuration_space.sample_configuration()
        run_history.add(config_3, 1, 1, StatusType.SUCCESS, seed=1,
                        instance_id='branin-hoo')
        # New config on no instance
        config_4 = configuration_space.sample_configuration()
        run_history.add(config_4, 1, 1, StatusType.SUCCESS, seed=1)

        # External configuration which will not be written to json file!
        config_5 = configuration_space.sample_configuration()
        run_history.add(config_5, 1, 1, StatusType.SUCCESS, seed=1,
                        origin=DataOrigin.EXTERNAL_SAME_INSTANCES)

        logger = logging.getLogger("Test")
        pSMAC.write(run_history, self.tmp_dir, logger=logger)
        r_size = len(run_history.data)
        pSMAC.read(run_history=run_history, 
                   output_dirs=[self.tmp_dir], 
                   configuration_space=configuration_space, 
                   logger=logger)
        self.assertEqual(r_size, len(run_history.data), 
                         "Runhistory should be the same and not changed after reading")
        

        output_filename = os.path.join(self.tmp_dir, 'runhistory.json')
        self.assertTrue(os.path.exists(output_filename))

        fixture = json.loads(fixture, object_hook=StatusType.enum_hook)
        with open(output_filename) as fh:
            output = json.load(fh, object_hook=StatusType.enum_hook)
        self.assertEqual(output, fixture)
        
    def test_load(self):
        configuration_space = test_helpers.get_branin_config_space()

        other_runhistory = '{"data": [[[2, "branini", 1], [1, 1,' \
                  '{"__enum__": "StatusType.SUCCESS"}, null]], ' \
                  '[[1, "branin", 1], [1, 1,' \
                  '{"__enum__": "StatusType.SUCCESS"}, null]], ' \
                  '[[3, "branin-hoo", 1], [1, 1,' \
                  '{"__enum__": "StatusType.SUCCESS"}, null]], ' \
                  '[[2, null, 1], [1, 1,' \
                  '{"__enum__": "StatusType.SUCCESS"}, null]], ' \
                  '[[1, "branini", 1], [1, 1,' \
                  '{"__enum__": "StatusType.SUCCESS"}, null]], ' \
                  '[[4, null, 1], [1, 1,' \
                  '{"__enum__": "StatusType.SUCCESS"}, null]]], ' \
                  '"configs": {' \
                  '"4": {"x": -2.2060968293349363, "y": 5.183410905645716}, ' \
                  '"3": {"x": -2.7986616377433045, "y": 1.385078921531967}, ' \
                  '"1": {"x": 1.2553300705386103, "y": 10.804867401632372}, ' \
                  '"2": {"x": -4.998284377739827, "y": 4.534988589477597}}}'

        other_runhistory_filename = os.path.join(self.tmp_dir,
                                                 'runhistory.json')
        with open(other_runhistory_filename, 'w') as fh:
            fh.write(other_runhistory)

        # load from an empty runhistory
        runhistory = RunHistory(aggregate_func=average_cost)
        runhistory.load_json(other_runhistory_filename, configuration_space)
        self.assertEqual(sorted(list(runhistory.ids_config.keys())),
                         [1, 2, 3, 4])
        self.assertEqual(len(runhistory.data), 6)

        # load from non-empty runhistory, in case of a duplicate the existing
        # result will be kept and the new one silently discarded
        runhistory = RunHistory(aggregate_func=average_cost)
        configuration_space.seed(1)
        config = configuration_space.sample_configuration()
        runhistory.add(config, 1, 1, StatusType.SUCCESS, seed=1,
                        instance_id='branin')
        id_before = id(runhistory.data[RunKey(1, 'branin', 1)])
        runhistory.update_from_json(other_runhistory_filename,
                                    configuration_space)
        id_after = id(runhistory.data[RunKey(1, 'branin', 1)])
        self.assertEqual(len(runhistory.data), 6)
        self.assertEqual(id_before, id_after)

        # load from non-empty runhistory, in case of a duplicate the existing
        # result will be kept and the new one silently discarded
        runhistory = RunHistory(aggregate_func=average_cost)
        configuration_space.seed(1)
        config = configuration_space.sample_configuration()
        config = configuration_space.sample_configuration()
        # This is the former config_3
        config = configuration_space.sample_configuration()
        runhistory.add(config, 1, 1, StatusType.SUCCESS, seed=1,
                       instance_id='branin')
        id_before = id(runhistory.data[RunKey(1, 'branin', 1)])
        runhistory.update_from_json(other_runhistory_filename,
                                    configuration_space)
        id_after = id(runhistory.data[RunKey(1, 'branin', 1)])
        self.assertEqual(len(runhistory.data), 7)
        self.assertEqual(id_before, id_after)
        self.assertEqual(sorted(list(runhistory.ids_config.keys())),
                         [1, 2, 3, 4])
        self.assertEqual([runhistory.external[run_key] for run_key in runhistory.data],
                         [DataOrigin.INTERNAL] + [DataOrigin.EXTERNAL_SAME_INSTANCES] * 6)
