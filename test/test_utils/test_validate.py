import unittest
import logging

import numpy as np

from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.optimizer.objective import average_cost
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

class ValidationTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.rng = np.random.RandomState(seed=42)
        self.scen = Scenario({'run_obj': 'quality', 'param_file':
                              'test/test_files/validation/test_validation_pcs.pcs',
                              'algo': 'python -u test/test_files/example_ta.py'})
        self.train_insts = ['0', '1', '2']
        self.test_insts = ['3', '4', '5']
        self.scen.instance_specific = {'0': 'null', '1': 'one', '2': 'two',
                                       '3': 'three', '4': 'four', '5': 'five'}
        self.output_rh = 'test/test_files/validation/test_validation_rh.json'
        self.trajectory = TrajLogger.read_traj_aclib_format(
            fn='test/test_files/validation/test_validation_traj.json', cs=self.scen.cs)

    def test_get_configs(self):
        validator = Validator(self.scen, self.trajectory,
                              self.output_rh, self.rng)
        self.assertEqual(1, len(validator._get_configs("def")))
        self.assertEqual(1, len(validator._get_configs("inc")))
        self.assertEqual(2, len(validator._get_configs("def+inc")))
        self.assertEqual(9, len(validator._get_configs("time")))
        self.assertEqual(9, len(validator._get_configs("all")))

    def test_get_runs(self):
        ''' test if the runs are generated as expected '''
        self.scen.train_insts = self.train_insts
        self.scen.test_insts = self.test_insts
        validator = Validator(self.scen, self.trajectory,
                              self.output_rh, self.rng)
        # Get multiple configs
        expected = [{'inst_specs': 'three', 'seed': 1608637542, 'inst': '3', 'config': 'config1'},
                    {'inst_specs': 'three', 'seed': 1608637542, 'inst': '3', 'config': 'config2'},
                    {'inst_specs': 'three', 'seed': 1273642419, 'inst': '3', 'config': 'config1'},
                    {'inst_specs': 'three', 'seed': 1273642419, 'inst': '3', 'config': 'config2'},
                    {'inst_specs': 'four',  'seed': 1935803228, 'inst': '4', 'config': 'config1'},
                    {'inst_specs': 'four',  'seed': 1935803228, 'inst': '4', 'config': 'config2'},
                    {'inst_specs': 'four',  'seed': 787846414,  'inst': '4', 'config': 'config1'},
                    {'inst_specs': 'four',  'seed': 787846414,  'inst': '4', 'config': 'config2'},
                    {'inst_specs': 'five',  'seed': 996406378,  'inst': '5', 'config': 'config1'},
                    {'inst_specs': 'five',  'seed': 996406378,  'inst': '5', 'config': 'config2'},
                    {'inst_specs': 'five',  'seed': 1201263687, 'inst': '5', 'config': 'config1'},
                    {'inst_specs': 'five',  'seed': 1201263687, 'inst': '5', 'config': 'config2'}]

        runs = validator.get_runs(['config1', 'config2'], self.scen.test_insts, repetitions=2)
        self.assertEqual(runs, expected)

        # Only train
        expected = [{'inst_specs': 'null', 'seed': 423734972,  'inst': '0', 'config': 'config1'},
                    {'inst_specs': 'null', 'seed': 415968276,  'inst': '0', 'config': 'config1'},
                    {'inst_specs': 'one',  'seed': 670094950,  'inst': '1', 'config': 'config1'},
                    {'inst_specs': 'one',  'seed': 1914837113, 'inst': '1', 'config': 'config1'},
                    {'inst_specs': 'two',  'seed': 669991378,  'inst': '2', 'config': 'config1'},
                    {'inst_specs': 'two',  'seed': 429389014,  'inst': '2', 'config': 'config1'}]

        runs = validator.get_runs(['config1'], self.scen.train_insts, repetitions=2)
        self.assertEqual(runs, expected)

        # Test and train
        expected = [{'inst': '0', 'seed': 249467210,  'config': 'config1', 'inst_specs': 'null'},
                    {'inst': '1', 'seed': 1972458954, 'config': 'config1', 'inst_specs': 'one'},
                    {'inst': '2', 'seed': 1572714583, 'config': 'config1', 'inst_specs': 'two'},
                    {'inst': '3', 'seed': 1433267572, 'config': 'config1', 'inst_specs': 'three'},
                    {'inst': '4', 'seed': 434285667,  'config': 'config1', 'inst_specs': 'four'},
                    {'inst': '5', 'seed': 613608295,  'config': 'config1', 'inst_specs': 'five'}]
        insts = self.train_insts
        insts.extend(self.test_insts)
        runs = validator.get_runs(['config1'], insts, repetitions=1)
        self.assertEqual(runs, expected)

    def test_validate(self):
        ''' test validation '''
        self.scen.train_insts = self.train_insts
        self.scen.test_insts = self.test_insts
        validator = Validator(self.scen, self.trajectory,
                              self.output_rh, self.rng)
        # Test basic usage
        rh = validator.validate(config_mode='def', instance_mode='test',
                                repetitions=3)
        self.assertEqual(len(rh.get_all_configs()), 1)
        self.assertEqual(len(rh.get_runs_for_config(rh.get_all_configs()[0])), 9)

        rh = validator.validate(config_mode='inc', instance_mode='train+test')
        self.assertEqual(len(rh.get_all_configs()), 1)
        self.assertEqual(len(rh.get_runs_for_config(rh.get_all_configs()[0])), 6)

        rh = validator.validate(config_mode='time', instance_mode='train')
        self.assertEqual(len(rh.get_all_configs()), 9)
        self.assertEqual(sum([len(rh.get_runs_for_config(c)) for c in
                              rh.get_all_configs()]), 27)

        # Test with backend multiprocessing
        rh = validator.validate(config_mode='def', instance_mode='test',
                                repetitions=3, backend='multiprocessing')
        self.assertEqual(len(rh.get_all_configs()), 1)
        self.assertEqual(len(rh.get_runs_for_config(rh.get_all_configs()[0])), 9)

    def test_validate_no_insts(self):
        ''' no instances '''
        validator = Validator(self.scen, self.trajectory,
                              self.output_rh, self.rng)
        rh = validator.validate(config_mode='def+inc',
                                instance_mode='train', repetitions=3)
        self.assertEqual(len(rh.get_all_configs()), 2)
        self.assertEqual(sum([len(rh.get_runs_for_config(c)) for c in
                              rh.get_all_configs()]), 6)

    def test_validate_deterministic(self):
        ''' deterministic ta '''
        self.scen.deterministic = True
        self.scen.train_insts = self.train_insts
        validator = Validator(self.scen, self.trajectory,
                              self.output_rh, self.rng)
        rh = validator.validate(config_mode='def+inc',
                                instance_mode='train', repetitions=3)
        self.assertEqual(len(rh.get_all_configs()), 2)
        self.assertEqual(sum([len(rh.get_runs_for_config(c)) for c in
                              rh.get_all_configs()]), 6)

    def test_parallel(self):
        ''' test parallel '''
        validator = Validator(self.scen, self.trajectory,
                              self.output_rh, self.rng)
        validator.validate(config_mode='all', instance_mode='train+test', n_jobs=-1)

    def test_passed_runhistory(self):
        ''' test if passed runhistory is in resulting runhistory '''
        self.scen.train_insts = self.train_insts
        self.scen.test_insts = self.test_insts
        validator = Validator(self.scen, self.trajectory,
                              self.output_rh, self.rng)
        # Add a few runs and check, if they are correctly processed
        old_configs = [entry["incumbent"] for entry in self.trajectory]
        old_rh = RunHistory(average_cost)
        for config in old_configs[:int(len(old_configs)/2)]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, instance_id='0',
                       seed=127)

        configs = validator._get_configs('all')
        insts = validator._get_instances('train')
        runs_w_rh = validator.get_runs(configs, insts, repetitions=2,
                                       runhistory=old_rh)
        runs_wo_rh = validator.get_runs(configs, insts, repetitions=2)
        self.assertEqual(len(runs_w_rh), len(runs_wo_rh) - 4)

    def test_passed_runhistory_deterministic(self):
        ''' test if passed runhistory is in resulting runhistory '''
        self.scen.deterministic = True
        self.scen.train_insts = self.train_insts
        validator = Validator(self.scen, self.trajectory,
                              self.output_rh, self.rng)
        # Add a few runs and check, if they are correctly processed
        old_configs = [entry["incumbent"] for entry in self.trajectory]
        old_rh = RunHistory(average_cost)
        for config in old_configs[:int(len(old_configs)/2)]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, instance_id='0')

        configs = validator._get_configs('all')
        insts = validator._get_instances('train')
        runs_w_rh = validator.get_runs(configs, insts, repetitions=2,
                                       runhistory=old_rh)
        runs_wo_rh = validator.get_runs(configs, insts, repetitions=2)
        self.assertEqual(len(runs_w_rh), len(runs_wo_rh) - 4)

    def test_passed_runhistory_no_insts(self):
        ''' test passed runhistory, without instances '''
        validator = Validator(self.scen, self.trajectory,
                              self.output_rh, self.rng)
        # Add a few runs and check, if they are correctly processed
        old_configs = [entry["incumbent"] for entry in self.trajectory]
        old_rh = RunHistory(average_cost)
        for config in old_configs[:int(len(old_configs)/2)]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, seed=127)

        configs = validator._get_configs('all')
        insts = validator._get_instances('train')
        runs_w_rh = validator.get_runs(configs, insts, repetitions=2,
                                       runhistory=old_rh)
        runs_wo_rh = validator.get_runs(configs, insts, repetitions=2)
        self.assertEqual(len(runs_w_rh), len(runs_wo_rh) - 4)
