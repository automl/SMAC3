import sys
import os
import unittest
from nose.plugins.attrib import attr
import logging
import shutil

import numpy as np

from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.optimizer.objective import average_cost
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator, _Run

from unittest import mock

class ValidationTest(unittest.TestCase):

    def setUp(self):
        base_directory = os.path.split(__file__)[0]
        base_directory = os.path.abspath(
            os.path.join(base_directory, '..', '..'))
        self.current_dir = os.getcwd()
        os.chdir(base_directory)

        logging.basicConfig()
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.rng = np.random.RandomState(seed=42)
        self.scen_fn = 'test/test_files/validation/scenario.txt'
        self.train_insts = ['0', '1', '2']
        self.test_insts = ['3', '4', '5']
        self.inst_specs = {'0': 'null', '1': 'one', '2': 'two',
                           '3': 'three', '4': 'four', '5': 'five'}
        self.feature_dict = {'0':np.array((1, 2, 3)),
                             '1':np.array((1, 2, 3)),
                             '2':np.array((1, 2, 3)),
                             '3':np.array((1, 2, 3)),
                             '4':np.array((1, 2, 3)),
                             '5':np.array((1, 2, 3))}
        self.output_rh = 'test/test_files/validation/'
        scen = Scenario(self.scen_fn, cmd_options={'run_obj': 'quality'})
        self.trajectory = TrajLogger.read_traj_aclib_format(
            fn='test/test_files/validation/test_validation_traj.json', cs=scen.cs)
        self.output_dirs = [self.output_rh + 'test']
        self.output_files = [self.output_rh + 'validated_runhistory_EPM.json', self.output_rh + 'validated_runhistory.json']

        self.maxDiff = None

    def tearDown(self):
        for output_dir in self.output_dirs:
            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)
        for output_file in self.output_files:
            if output_file:
                try:
                    os.remove(output_file)
                except FileNotFoundError as e:
                    pass
        os.chdir(self.current_dir)

    def test_rng(self):
        scen = Scenario(self.scen_fn, cmd_options={'run_obj': 'quality'})
        validator = Validator(scen, self.trajectory, 42)
        self.assertTrue(isinstance(validator.rng, np.random.RandomState))
        validator = Validator(scen, self.trajectory)
        self.assertTrue(isinstance(validator.rng, np.random.RandomState))
        validator = Validator(scen, self.trajectory, np.random.RandomState())
        self.assertTrue(isinstance(validator.rng, np.random.RandomState))

    def test_nonexisting_output(self):
        scen = Scenario(self.scen_fn, cmd_options={'run_obj': 'quality'})
        validator = Validator(scen, self.trajectory)
        path = "test/test_files/validation/test/nonexisting/output"
        validator.validate(output_fn=path)
        self.assertTrue(os.path.exists(path))

    def test_pass_tae(self):
        scen = Scenario(self.scen_fn, cmd_options={'run_obj': 'quality'})
        tae = ExecuteTARunOld(ta=scen.ta)
        validator = Validator(scen, self.trajectory)
        with mock.patch.object(Validator, "_validate_parallel",
                               return_value=[(1,2,3,4)]):
            self.assertEqual(1, len(validator.validate(tae=tae).data))

    def test_no_rh_epm(self):
        scen = Scenario(self.scen_fn, cmd_options={'run_obj': 'quality'})
        scen.feature_array = None
        validator = Validator(scen, self.trajectory)
        self.assertRaises(ValueError, validator.validate_epm)

    def test_epm_reuse_rf(self):
        """ if no runhistory is passed to epm, but there was a model trained
        before, that model should be reused! (if reuse_epm flag is set) """
        scen = Scenario(self.scen_fn, cmd_options={'run_obj': 'quality'})
        scen.feature_array = None
        validator = Validator(scen, self.trajectory)
        old_rh = RunHistory(average_cost)
        for config in [e["incumbent"] for e in self.trajectory]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, instance_id='0',
                       seed=127)
        self.assertTrue(isinstance(validator.validate_epm(runhistory=old_rh),
                                   RunHistory))
        self.assertTrue(isinstance(validator.validate_epm(
                                    output_fn="test/test_files/validation/"),
                                    RunHistory))
        self.assertRaises(ValueError, validator.validate_epm, reuse_epm=False)

    def test_no_feature_dict(self):
        scen = Scenario(self.scen_fn, cmd_options={'run_obj': 'quality'})
        scen.feature_array = None
        validator = Validator(scen, self.trajectory)
        old_rh = RunHistory(average_cost)
        for config in [e["incumbent"] for e in self.trajectory]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, instance_id='0',
                       seed=127)
        validator.validate_epm(runhistory=old_rh)

    def test_get_configs(self):
        scen = Scenario(self.scen_fn, cmd_options={'run_obj': 'quality'})
        validator = Validator(scen, self.trajectory, self.rng)
        self.assertEqual(1, len(validator._get_configs("def")))
        self.assertEqual(1, len(validator._get_configs("inc")))
        self.assertEqual(2, len(validator._get_configs("def+inc")))
        self.assertEqual(7, len(validator._get_configs("wallclock_time")))
        self.assertEqual(8, len(validator._get_configs("cpu_time")))
        self.assertEqual(10, len(validator._get_configs("all")))
        # Using maxtime
        validator.scen.wallclock_limit = 65
        validator.scen.algo_runs_timelimit = 33
        self.assertEqual(8, len(validator._get_configs("wallclock_time")))
        self.assertEqual(9, len(validator._get_configs("cpu_time")))
        # Exceptions
        self.assertRaises(ValueError, validator._get_configs, "notanoption")
        self.assertRaises(ValueError, validator._get_instances, "notanoption")

    def test_get_runs_capped(self):
        ''' test if capped, crashed and aborted runs are ignored
            during rh-recovery '''
        scen = Scenario(self.scen_fn,
                        cmd_options={'run_obj':'quality',
                                     'instances' : ['0']})

        validator = Validator(scen, self.trajectory, self.rng)

        # Get runhistory
        old_configs = ['config1', 'config2', 'config3',
                       'config4', 'config5', 'config6']
        old_rh = RunHistory(average_cost)
        old_rh.add('config1', 1, 1, StatusType.SUCCESS, instance_id='0', seed=0)
        old_rh.add('config2', 1, 1, StatusType.TIMEOUT, instance_id='0', seed=0)
        old_rh.add('config3', 1, 1, StatusType.CRASHED, instance_id='0', seed=0)
        old_rh.add('config4', 1, 1, StatusType.ABORT, instance_id='0', seed=0)
        old_rh.add('config5', 1, 1, StatusType.MEMOUT, instance_id='0', seed=0)
        old_rh.add('config6', 1, 1, StatusType.CAPPED, instance_id='0', seed=0)

        # Get multiple configs
        expected = [_Run(inst_specs='0', seed=0, inst='0', config='config3'),
                    _Run(inst_specs='0', seed=0, inst='0', config='config4'),
                    _Run(inst_specs='0', seed=0, inst='0', config='config6')]

        runs = validator._get_runs(old_configs, ['0'], repetitions=1, runhistory=old_rh)
        self.assertEqual(runs[0], expected)

    def test_get_runs(self):
        ''' test if the runs are generated as expected '''
        scen = Scenario(self.scen_fn,
                        cmd_options={'run_obj': 'quality',
                                     'train_insts' : self.train_insts,
                                     'test_insts': self.test_insts})
        scen.instance_specific = self.inst_specs

        validator = Validator(scen, self.trajectory, self.rng)
        # Get multiple configs
        self.maxDiff=None
        expected = [_Run(config='config1', inst='3', seed=1608637542, inst_specs='three'),
                    _Run(config='config2', inst='3', seed=1608637542, inst_specs='three'),
                    _Run(config='config1', inst='3', seed=1273642419, inst_specs='three'),
                    _Run(config='config2', inst='3', seed=1273642419, inst_specs='three'),
                    _Run(config='config1', inst='4', seed=1935803228, inst_specs='four'),
                    _Run(config='config2', inst='4', seed=1935803228, inst_specs='four'),
                    _Run(config='config1', inst='4', seed=787846414, inst_specs='four'),
                    _Run(config='config2', inst='4', seed=787846414, inst_specs='four'),
                    _Run(config='config1', inst='5', seed=996406378, inst_specs='five'),
                    _Run(config='config2', inst='5', seed=996406378, inst_specs='five'),
                    _Run(config='config1', inst='5', seed=1201263687, inst_specs='five'),
                    _Run(config='config2', inst='5', seed=1201263687, inst_specs='five')]

        runs = validator._get_runs(['config1', 'config2'], scen.test_insts, repetitions=2)
        self.assertEqual(runs[0], expected)

        # Only train
        expected = [_Run(config='config1', inst='0', seed=423734972, inst_specs='null'),
                    _Run(config='config1', inst='0', seed=415968276, inst_specs='null'),
                    _Run(config='config1', inst='1', seed=670094950, inst_specs='one'),
                    _Run(config='config1', inst='1', seed=1914837113, inst_specs='one'),
                    _Run(config='config1', inst='2', seed=669991378, inst_specs='two'),
                    _Run(config='config1', inst='2', seed=429389014, inst_specs='two')]

        runs = validator._get_runs(['config1'], scen.train_insts, repetitions=2)
        self.assertEqual(runs[0], expected)

        # Test and train
        expected = [_Run(config='config1', inst='0', seed=249467210, inst_specs='null'),
                    _Run(config='config1', inst='1', seed=1972458954, inst_specs='one'),
                    _Run(config='config1', inst='2', seed=1572714583, inst_specs='two'),
                    _Run(config='config1', inst='3', seed=1433267572, inst_specs='three'),
                    _Run(config='config1', inst='4', seed=434285667, inst_specs='four'),
                    _Run(config='config1', inst='5', seed=613608295, inst_specs='five')]

        insts = self.train_insts
        insts.extend(self.test_insts)
        runs = validator._get_runs(['config1'], insts, repetitions=1)
        self.assertEqual(runs[0], expected)

    @attr('slow')
    def test_validate(self):
        ''' test validation '''
        scen = Scenario(self.scen_fn,
                        cmd_options={'run_obj': 'quality',
                                  'train_insts' : self.train_insts,
                                  'test_insts': self.test_insts})
        scen.instance_specific = self.inst_specs
        validator = Validator(scen, self.trajectory, self.rng)
        # Test basic usage
        rh = validator.validate(config_mode='def', instance_mode='test',
                                repetitions=3)
        self.assertEqual(len(rh.get_all_configs()), 1)
        self.assertEqual(len(rh.get_runs_for_config(rh.get_all_configs()[0])), 9)

        rh = validator.validate(config_mode='inc', instance_mode='train+test')
        self.assertEqual(len(rh.get_all_configs()), 1)
        self.assertEqual(len(rh.get_runs_for_config(rh.get_all_configs()[0])), 6)

        rh = validator.validate(config_mode='wallclock_time', instance_mode='train')
        self.assertEqual(len(rh.get_all_configs()), 7)
        self.assertEqual(sum([len(rh.get_runs_for_config(c)) for c in
                              rh.get_all_configs()]), 21)

        # Test with backend multiprocessing
        rh = validator.validate(config_mode='def', instance_mode='test',
                                repetitions=3, backend='multiprocessing')
        self.assertEqual(len(rh.get_all_configs()), 1)
        self.assertEqual(len(rh.get_runs_for_config(rh.get_all_configs()[0])), 9)

    @attr('slow')
    def test_validate_no_insts(self):
        ''' no instances '''
        scen = Scenario(self.scen_fn,
                        cmd_options={'run_obj': 'quality'})
        validator = Validator(scen, self.trajectory, self.rng)
        rh = validator.validate(config_mode='def+inc', instance_mode='train',
                                repetitions=3, output_fn=self.output_rh)
        self.assertEqual(len(rh.get_all_configs()), 2)
        self.assertEqual(sum([len(rh.get_runs_for_config(c)) for c in
                              rh.get_all_configs()]), 6)

    @attr('slow')
    def test_validate_deterministic(self):
        ''' deterministic ta '''
        scen = Scenario(self.scen_fn,
                        cmd_options={'run_obj': 'quality',
                                  'train_insts' : self.train_insts,
                                  'deterministic': True})
        scen.instance_specific = self.inst_specs
        validator = Validator(scen, self.trajectory, self.rng)
        rh = validator.validate(config_mode='def+inc',
                                instance_mode='train', repetitions=3)
        self.assertEqual(len(rh.get_all_configs()), 2)
        self.assertEqual(sum([len(rh.get_runs_for_config(c)) for c in
                              rh.get_all_configs()]), 6)

    @attr('slow')
    def test_parallel(self):
        ''' test parallel '''
        scen = Scenario(self.scen_fn,
                        cmd_options={'run_obj': 'quality'})
        validator = Validator(scen, self.trajectory, self.rng)
        validator.validate(config_mode='all', instance_mode='train+test', n_jobs=-1)

    def test_passed_runhistory(self):
        ''' test if passed runhistory is in resulting runhistory '''
        scen = Scenario(self.scen_fn,
                        cmd_options={'run_obj': 'quality',
                                  'train_insts' : self.train_insts,
                                  'test_insts': self.test_insts})
        scen.instance_specific = self.inst_specs
        validator = Validator(scen, self.trajectory, self.rng)
        # Add a few runs and check, if they are correctly processed
        old_configs = [entry["incumbent"] for entry in self.trajectory]
        old_rh = RunHistory(average_cost)
        seeds = [127 for i in range(int(len(old_configs)/2))]
        seeds[-1] = 126  # Test instance_seed-structure in validation
        for config in old_configs[:int(len(old_configs)/2)]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, instance_id='0',
                       seed=seeds[old_configs.index(config)])

        configs = validator._get_configs('all')
        insts = validator._get_instances('train')
        runs_w_rh = validator._get_runs(configs, insts, repetitions=2,
                                       runhistory=old_rh)
        runs_wo_rh = validator._get_runs(configs, insts, repetitions=2)
        self.assertEqual(len(runs_w_rh[0]), len(runs_wo_rh[0]) - 4)
        self.assertEqual(len(runs_w_rh[1].data), 4)
        self.assertEqual(len(runs_wo_rh[1].data), 0)

    def test_passed_runhistory_deterministic(self):
        ''' test if passed runhistory is in resulting runhistory '''
        scen = Scenario(self.scen_fn,
                        cmd_options={'run_obj': 'quality',
                                  'train_insts' : self.train_insts,
                                  'deterministic' : True})
        scen.instance_specific = self.inst_specs
        validator = Validator(scen, self.trajectory, self.rng)
        # Add a few runs and check, if they are correctly processed
        old_configs = [entry["incumbent"] for entry in self.trajectory]
        old_rh = RunHistory(average_cost)
        for config in old_configs[:int(len(old_configs)/2)]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, instance_id='0')

        configs = validator._get_configs('all')
        insts = validator._get_instances('train')
        runs_w_rh = validator._get_runs(configs, insts, repetitions=2,
                                       runhistory=old_rh)
        runs_wo_rh = validator._get_runs(configs, insts, repetitions=2)
        self.assertEqual(len(runs_w_rh[0]), len(runs_wo_rh[0]) - 4)
        self.assertEqual(len(runs_w_rh[1].data), 4)
        self.assertEqual(len(runs_wo_rh[1].data), 0)

    def test_passed_runhistory_no_insts(self):
        ''' test passed runhistory, without instances '''
        scen = Scenario(self.scen_fn,
                        cmd_options={'run_obj': 'quality'})
        scen.instance_specific = self.inst_specs
        validator = Validator(scen, self.trajectory, self.rng)
        # Add a few runs and check, if they are correctly processed
        old_configs = [entry["incumbent"] for entry in self.trajectory]
        old_rh = RunHistory(average_cost)
        for config in old_configs[:int(len(old_configs)/2)]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, seed=127)

        configs = validator._get_configs('all')
        insts = validator._get_instances('train')
        runs_w_rh = validator._get_runs(configs, insts, repetitions=2,
                                       runhistory=old_rh)
        runs_wo_rh = validator._get_runs(configs, insts, repetitions=2)
        self.assertEqual(len(runs_w_rh[0]), len(runs_wo_rh[0]) - 4)
        self.assertEqual(len(runs_w_rh[1].data), 4)
        self.assertEqual(len(runs_wo_rh[1].data), 0)

    def test_validate_epm(self):
        ''' test using epm to validate '''
        scen = Scenario(self.scen_fn,
                        cmd_options={'run_obj': 'quality',
                                  'train_insts' : self.train_insts,
                                  'test_insts': self.test_insts,
                                  'features': self.feature_dict})
        scen.instance_specific = self.inst_specs
        validator = Validator(scen, self.trajectory, self.rng)
        # Add a few runs and check, if they are correctly processed
        old_configs = [entry["incumbent"] for entry in self.trajectory]
        old_rh = RunHistory(average_cost)
        for config in old_configs[:int(len(old_configs)/2)]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, instance_id='0',
                       seed=127)
        validator.validate_epm('all', 'train', 1, old_rh)

    def test_objective_runtime(self):
        ''' test if everything is ok with objective runtime (imputing!) '''
        scen = Scenario(self.scen_fn, cmd_options={'run_obj' : 'runtime',
                                                   'cutoff_time' : 5})
        validator = Validator(scen, self.trajectory, self.rng)
        old_configs = [entry["incumbent"] for entry in self.trajectory]
        old_rh = RunHistory(average_cost)
        for config in old_configs[:int(len(old_configs)/2)]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, instance_id='0')
        validator.validate_epm('all', 'train', 1, old_rh)

    def test_inst_no_feat(self):
        ''' test if scenarios are treated correctly if no features are
        specified.'''
        scen = Scenario(self.scen_fn,
                        cmd_options={'run_obj': 'quality',
                                  'train_insts' : self.train_insts,
                                  'test_insts': self.test_insts})
        self.assertTrue(scen.feature_array is None)
        self.assertEqual(len(scen.feature_dict), 0)

        scen.instance_specific = self.inst_specs
        validator = Validator(scen, self.trajectory, self.rng)
        # Add a few runs and check, if they are correctly processed
        old_configs = [entry["incumbent"] for entry in self.trajectory]
        old_rh = RunHistory(average_cost)
        for config in old_configs[:int(len(old_configs)/2)]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, instance_id='0',
                       seed=127)
        rh = validator.validate_epm('all', 'train+test', 1, old_rh)
        self.assertEqual(len(old_rh.get_all_configs()), 4)
        self.assertEqual(len(rh.get_all_configs()), 10)
