import sys
import os
import unittest
import logging

import numpy as np

from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.optimizer.objective import average_cost
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator, Run

from unittest import mock

class ValidationTest(unittest.TestCase):

    def setUp(self):
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
        scen = Scenario(self.scen_fn, cmd_args={'run_obj':'quality'})
        self.trajectory = TrajLogger.read_traj_aclib_format(
            fn='test/test_files/validation/test_validation_traj.json', cs=scen.cs)

    def test_rng(self):
        scen = Scenario(self.scen_fn, cmd_args={'run_obj':'quality'})
        validator = Validator(scen, self.trajectory, 42)
        self.assertTrue(isinstance(validator.rng, np.random.RandomState))
        validator = Validator(scen, self.trajectory)
        self.assertTrue(isinstance(validator.rng, np.random.RandomState))
        validator = Validator(scen, self.trajectory, np.random.RandomState())
        self.assertTrue(isinstance(validator.rng, np.random.RandomState))

    def test_nonexisting_output(self):
        scen = Scenario(self.scen_fn, cmd_args={'run_obj':'quality'})
        validator = Validator(scen, self.trajectory)
        path = "test/test_files/validation/test/nonexisting/output"
        validator.validate(output_fn=path)
        self.assertTrue(os.path.exists(path))

    def test_pass_tae(self):
        scen = Scenario(self.scen_fn, cmd_args={'run_obj':'quality'})
        tae = ExecuteTARunOld(ta=scen.ta)
        validator = Validator(scen, self.trajectory)
        with mock.patch.object(Validator, "_validate_parallel",
                               return_value=[(1,2,3,4)]):
            self.assertEqual(1, len(validator.validate(tae=tae).data))

    def test_no_rh_epm(self):
        scen = Scenario(self.scen_fn, cmd_args={'run_obj':'quality'})
        scen.feature_array = None
        validator = Validator(scen, self.trajectory)
        self.assertRaises(ValueError, validator.validate_epm)

    def test_epm_reuse_rf(self):
        """ if no runhistory is passed to epm, but there was a model trained
        before, that model should be reused! (if reuse_epm flag is set) """
        scen = Scenario(self.scen_fn, cmd_args={'run_obj':'quality'})
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
        scen = Scenario(self.scen_fn, cmd_args={'run_obj':'quality'})
        scen.feature_array = None
        validator = Validator(scen, self.trajectory)
        old_rh = RunHistory(average_cost)
        for config in [e["incumbent"] for e in self.trajectory]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, instance_id='0',
                       seed=127)
        validator.validate_epm(runhistory=old_rh)

    def test_get_configs(self):
        scen = Scenario(self.scen_fn, cmd_args={'run_obj':'quality'})
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

    def test_get_runs(self):
        ''' test if the runs are generated as expected '''
        scen = Scenario(self.scen_fn,
                        cmd_args={'run_obj':'quality',
                                  'instances' : self.train_insts,
                                  'test_instances': self.test_insts})
        scen.instance_specific = self.inst_specs

        validator = Validator(scen, self.trajectory, self.rng)
        # Get multiple configs
        expected = [Run(inst_specs='three', seed=1608637542, inst='3', config='config1'),
                    Run(inst_specs='three', seed=1608637542, inst='3', config='config2'),
                    Run(inst_specs='three', seed=1935803228, inst='3', config='config1'),
                    Run(inst_specs='three', seed=1935803228, inst='3', config='config2'),
                    Run(inst_specs='four',  seed=996406378, inst='4', config='config1'),
                    Run(inst_specs='four',  seed=996406378, inst='4', config='config2'),
                    Run(inst_specs='four',  seed=423734972,  inst='4', config='config1'),
                    Run(inst_specs='four',  seed=423734972,  inst='4', config='config2'),
                    Run(inst_specs='five',  seed=670094950,  inst='5', config='config1'),
                    Run(inst_specs='five',  seed=670094950,  inst='5', config='config2'),
                    Run(inst_specs='five',  seed=669991378, inst='5', config='config1'),
                    Run(inst_specs='five',  seed=669991378, inst='5', config='config2')]

        runs = validator.get_runs(['config1', 'config2'], scen.test_insts, repetitions=2)
        self.assertEqual(runs[0], expected)

        # Only train
        expected = [Run(inst_specs='null', seed=249467210,  inst='0', config='config1'),
                    Run(inst_specs='null', seed=1572714583,  inst='0', config='config1'),
                    Run(inst_specs='one',  seed=434285667,  inst='1', config='config1'),
                    Run(inst_specs='one',  seed=893664919, inst='1', config='config1'),
                    Run(inst_specs='two',  seed=88409749,  inst='2', config='config1'),
                    Run(inst_specs='two',  seed=2018247425,  inst='2', config='config1')]

        runs = validator.get_runs(['config1'], scen.train_insts, repetitions=2)
        self.assertEqual(runs[0], expected)

        # Test and train
        expected = [Run(inst='0', seed=1427830251,  config='config1', inst_specs='null' ),
                    Run(inst='1', seed=911989541, config='config1', inst_specs='one'  ),
                    Run(inst='2', seed=780932287, config='config1', inst_specs='two'  ),
                    Run(inst='3', seed=787716372, config='config1', inst_specs='three'),
                    Run(inst='4', seed=1306710475,  config='config1', inst_specs='four' ),
                    Run(inst='5', seed=106328085,  config='config1', inst_specs='five' )]
        insts = self.train_insts
        insts.extend(self.test_insts)
        runs = validator.get_runs(['config1'], insts, repetitions=1)
        self.assertEqual(runs[0], expected)

    def test_validate(self):
        ''' test validation '''
        scen = Scenario(self.scen_fn,
                        cmd_args={'run_obj':'quality',
                                  'instances' : self.train_insts,
                                  'test_instances': self.test_insts})
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

    def test_validate_no_insts(self):
        ''' no instances '''
        scen = Scenario(self.scen_fn,
                        cmd_args={'run_obj':'quality'})
        validator = Validator(scen, self.trajectory, self.rng)
        rh = validator.validate(config_mode='def+inc', instance_mode='train',
                                repetitions=3, output_fn=self.output_rh)
        self.assertEqual(len(rh.get_all_configs()), 2)
        self.assertEqual(sum([len(rh.get_runs_for_config(c)) for c in
                              rh.get_all_configs()]), 6)

    def test_validate_deterministic(self):
        ''' deterministic ta '''
        scen = Scenario(self.scen_fn,
                        cmd_args={'run_obj':'quality',
                                  'instances' : self.train_insts,
                                  'deterministic': True})
        scen.instance_specific = self.inst_specs
        validator = Validator(scen, self.trajectory, self.rng)
        rh = validator.validate(config_mode='def+inc',
                                instance_mode='train', repetitions=3)
        self.assertEqual(len(rh.get_all_configs()), 2)
        self.assertEqual(sum([len(rh.get_runs_for_config(c)) for c in
                              rh.get_all_configs()]), 6)

    def test_parallel(self):
        ''' test parallel '''
        scen = Scenario(self.scen_fn,
                        cmd_args={'run_obj':'quality'})
        validator = Validator(scen, self.trajectory, self.rng)
        validator.validate(config_mode='all', instance_mode='train+test', n_jobs=-1)

    def test_passed_runhistory(self):
        ''' test if passed runhistory is in resulting runhistory '''
        scen = Scenario(self.scen_fn,
                        cmd_args={'run_obj':'quality',
                                  'instances' : self.train_insts,
                                  'test_instances': self.test_insts})
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
        runs_w_rh = validator.get_runs(configs, insts, repetitions=2,
                                       runhistory=old_rh)
        runs_wo_rh = validator.get_runs(configs, insts, repetitions=2)
        self.assertEqual(len(runs_w_rh[0]), len(runs_wo_rh[0]) - 4)
        self.assertEqual(len(runs_w_rh[1].data), 4)
        self.assertEqual(len(runs_wo_rh[1].data), 0)

    def test_passed_runhistory_deterministic(self):
        ''' test if passed runhistory is in resulting runhistory '''
        scen = Scenario(self.scen_fn,
                        cmd_args={'run_obj':'quality',
                                  'instances' : self.train_insts,
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
        runs_w_rh = validator.get_runs(configs, insts, repetitions=2,
                                       runhistory=old_rh)
        runs_wo_rh = validator.get_runs(configs, insts, repetitions=2)
        self.assertEqual(len(runs_w_rh[0]), len(runs_wo_rh[0]) - 4)
        self.assertEqual(len(runs_w_rh[1].data), 4)
        self.assertEqual(len(runs_wo_rh[1].data), 0)

    def test_passed_runhistory_no_insts(self):
        ''' test passed runhistory, without instances '''
        scen = Scenario(self.scen_fn,
                        cmd_args={'run_obj':'quality'})
        scen.instance_specific = self.inst_specs
        validator = Validator(scen, self.trajectory, self.rng)
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
        self.assertEqual(len(runs_w_rh[0]), len(runs_wo_rh[0]) - 4)
        self.assertEqual(len(runs_w_rh[1].data), 4)
        self.assertEqual(len(runs_wo_rh[1].data), 0)

    def test_validate_epm(self):
        ''' test using epm to validate '''
        scen = Scenario(self.scen_fn,
                        cmd_args={'run_obj':'quality',
                                  'instances' : self.train_insts,
                                  'test_instances': self.test_insts,
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
        scen = Scenario(self.scen_fn, cmd_args={'run_obj' : 'runtime',
                                                'cutoff_time' : 5})
        validator = Validator(scen, self.trajectory, self.rng)
        old_configs = [entry["incumbent"] for entry in self.trajectory]
        old_rh = RunHistory(average_cost)
        for config in old_configs[:int(len(old_configs)/2)]:
            old_rh.add(config, 1, 1, StatusType.SUCCESS, instance_id='0')
        validator.validate_epm('all', 'train', 1, old_rh)
