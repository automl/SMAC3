import unittest
from nose.plugins.attrib import attr

import logging
import numpy as np
import time

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.intensification.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory, RunKey
from smac.optimizer.objective import average_cost, sum_cost
from smac.tae.execute_ta_run import StatusType
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger


def get_config_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='a',
                                                       lower=0,
                                                       upper=100))
    cs.add_hyperparameter(UniformIntegerHyperparameter(name='b',
                                                       lower=0,
                                                       upper=100))
    return cs


class TestIntensify(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        self.rh = RunHistory(aggregate_func=average_cost)
        self.cs = get_config_space()
        self.config1 = Configuration(self.cs,
                                     values={'a': 0, 'b': 100})
        self.config2 = Configuration(self.cs,
                                     values={'a': 100, 'b': 0})
        self.config3 = Configuration(self.cs,
                                     values={'a': 100, 'b': 100})

        self.scen = Scenario({"cutoff_time": 2, 'cs': self.cs,
                              "run_obj": 'runtime',
                              "output_dir": ''})
        self.stats = Stats(scenario=self.scen)
        self.stats.start_timing()

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

    def test_compare_configs_no_joint_set(self):
        intensifier = Intensifier(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=None, instances=[1])

        for i in range(2):
            self.rh.add(config=self.config1, cost=2, time=2,
                        status=StatusType.SUCCESS, instance_id=1,
                        seed=i, additional_info=None)

        for i in range(2, 5):
            self.rh.add(config=self.config2, cost=1, time=1,
                        status=StatusType.SUCCESS, instance_id=1,
                        seed=i, additional_info=None)

        # The sets for the incumbent are completely disjoint.
        conf = intensifier._compare_configs(incumbent=self.config1,
                                            challenger=self.config2,
                                            run_history=self.rh,
                                            aggregate_func=average_cost)
        self.assertIsNone(conf)

        # The incumbent has still one instance-seed pair left on which the
        # challenger was not run yet.
        self.rh.add(config=self.config2, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=1, additional_info=None)
        conf = intensifier._compare_configs(incumbent=self.config1,
                                            challenger=self.config2,
                                            run_history=self.rh,
                                            aggregate_func=average_cost)
        self.assertIsNone(conf)

    def test_compare_configs_chall(self):
        '''
            challenger is better
        '''
        intensifier = Intensifier(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=None,
            instances=[1])

        self.rh.add(config=self.config1, cost=1, time=2,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config2, cost=0, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        conf = intensifier._compare_configs(incumbent=self.config1,
                                            challenger=self.config2,
                                            run_history=self.rh,
                                            aggregate_func=average_cost)

        # challenger has enough runs and is better
        self.assertEqual(conf, self.config2, "conf: %s" % (conf))

    def test_compare_configs_inc(self):
        '''
            incumbent is better
        '''
        intensifier = Intensifier(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=None,
            instances=[1])

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config2, cost=2, time=2,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        conf = intensifier._compare_configs(incumbent=self.config1,
                                            challenger=self.config2,
                                            run_history=self.rh,
                                            aggregate_func=average_cost)

        # challenger worse than inc
        self.assertEqual(conf, self.config1, "conf: %s" % (conf))

    def test_compare_configs_unknow(self):
        '''
            challenger is better but has less runs;
            -> no decision (None)
        '''
        intensifier = Intensifier(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=None,
            instances=[1])

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config1, cost=1, time=2,
                    status=StatusType.SUCCESS, instance_id=2,
                    seed=None,
                    additional_info=None)

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=2,
                    seed=None,
                    additional_info=None)

        conf = intensifier._compare_configs(incumbent=self.config1,
                                            challenger=self.config2,
                                            run_history=self.rh,
                                            aggregate_func=average_cost)

        # challenger worse than inc
        self.assertIsNone(conf, "conf: %s" % (conf))

    @attr('slow')
    def test_race_challenger(self):
        '''
           test _race_challenger without adaptive capping
        '''

        def target(x):
            return (x['a'] + 1) / 1000.
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1])

        self.rh.add(config=self.config1, cost=1, time=1,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=None,
                    additional_info=None)

        inc = intensifier._race_challenger(challenger=self.config2,
                                           incumbent=self.config1,
                                           run_history=self.rh,
                                           aggregate_func=average_cost)

        self.assertEqual(inc, self.config2)

    @attr('slow')
    def test_race_challenger_2(self):
        '''
           test _race_challenger with adaptive capping
        '''

        def target(x):
            time.sleep(1.5)
            return (x['a'] + 1) / 1000.
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime")
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1])

        self.rh.add(config=self.config1, cost=.001, time=0.001,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=12345,
                    additional_info=None)

        # config2 should have a timeout (due to adaptive capping)
        # and config1 should still be the incumbent
        inc = intensifier._race_challenger(challenger=self.config2,
                                           incumbent=self.config1,
                                           run_history=self.rh,
                                           aggregate_func=average_cost)

        # self.assertTrue(False)
        self.assertEqual(inc, self.config1)

    @attr('slow')
    def test_race_challenger_3(self):
        '''
           test _race_challenger with adaptive capping on a previously capped configuration  
        '''

        def target(config: Configuration, seed: int, instance: str):
            if instance == 1:
                time.sleep(2.1)
            else:
                time.sleep(0.6)
            return (config['a'] + 1) / 1000.
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="runtime", par_factor=1)
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            cutoff=2,
            instances=[1])

        self.rh.add(config=self.config1, cost=0.5, time=.5,
                    status=StatusType.SUCCESS, instance_id=1,
                    seed=12345,
                    additional_info=None)
        
        # config2 should have a timeout (due to adaptive capping)
        # and config1 should still be the incumbent
        inc = intensifier._race_challenger(challenger=self.config2,
                                           incumbent=self.config1,
                                           run_history=self.rh,
                                           aggregate_func=average_cost)
        # self.assertTrue(False)
        self.assertEqual(inc, self.config1)
        
        # further run for incumbent
        self.rh.add(config=self.config1, cost=2, time=2,
                    status=StatusType.TIMEOUT, instance_id=2,
                    seed=12345,
                    additional_info=None)
        
        # give config2 a second chance
        inc = intensifier._race_challenger(challenger=self.config2,
                               incumbent=self.config1,
                               run_history=self.rh,
                               aggregate_func=average_cost)      
        
        # the incumbent should still be config1 because
        # config2 should get on inst 1 a full timeout
        # such that c(config1) = 1.25 and c(config2) close to 1.3
        self.assertEqual(inc, self.config1)
        # the capped run should not be counted in runs_perf_config
        self.assertAlmostEqual(self.rh.runs_per_config[2], 2)

    def test_race_challenger_large(self):
        '''
           test _race_challenger using solution_quality
        '''

        def target(x):
            return 1

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=list(range(10)),
            deterministic=True)

        for i in range(10):
            self.rh.add(config=self.config1, cost=i+1, time=1,
                    status=StatusType.SUCCESS, instance_id=i,
                    seed=12345,
                    additional_info=None)

        # tie on first instances and then challenger should always win
        # and be returned as inc
        inc = intensifier._race_challenger(challenger=self.config2,
                                           incumbent=self.config1,
                                           run_history=self.rh,
                                           aggregate_func=average_cost)

        # self.assertTrue(False)
        self.assertEqual(inc, self.config2)
        self.assertEqual(
            self.rh.get_cost(self.config2), 1, self.rh.get_cost(self.config2))


        # get data for config2 to check that the correct run was performed
        runs = self.rh.get_runs_for_config(self.config2)
        self.assertEqual(len(runs),10)

    def test_race_challenger_large_blocked_seed(self):
        '''
           test _race_challenger whether seeds are blocked for challenger runs
        '''

        def target(x):
            return 1

        taf = ExecuteTAFuncDict(ta=target, stats=self.stats)
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=list(range(10)),
            deterministic=False)

        for i in range(10):
            self.rh.add(config=self.config1, cost=i+1, time=1,
                    status=StatusType.SUCCESS, instance_id=i,
                    seed=i,
                    additional_info=None)

        # tie on first instances and then challenger should always win
        # and be returned as inc
        inc = intensifier._race_challenger(challenger=self.config2,
                                           incumbent=self.config1,
                                           run_history=self.rh,
                                           aggregate_func=average_cost)

        # self.assertTrue(False)
        self.assertEqual(inc, self.config2)
        self.assertEqual(
            self.rh.get_cost(self.config2), 1, self.rh.get_cost(self.config2))

        # get data for config2 to check that the correct run was performed
        runs = self.rh.get_runs_for_config(self.config2)
        self.assertEqual(len(runs), 10)

        seeds = sorted([r.seed for r in runs])
        self.assertEqual(seeds, list(range(10)), seeds)

    def test_add_inc_run_det(self):
        '''
            test _add_inc_run()
        '''

        def target(x):
            return (x['a'] + 1) / 1000.
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="solution_quality")
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1],
            deterministic=True)

        intensifier._add_inc_run(incumbent=self.config1, run_history=self.rh)
        self.assertEqual(len(self.rh.data), 1, self.rh.data)

        # since we assume deterministic=1,
        # the second call should not add any more runs
        # given only one instance
        intensifier._add_inc_run(incumbent=self.config1, run_history=self.rh)
        self.assertEqual(len(self.rh.data), 1, self.rh.data)

    def test_add_inc_run_nondet(self):
        '''
            test _add_inc_run()
        '''

        def target(x):
            return (x['a'] + 1) / 1000.
        taf = ExecuteTAFuncDict(ta=target, stats=self.stats, run_obj="solution_quality")
        taf.runhistory = self.rh

        intensifier = Intensifier(
            tae_runner=taf, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=[1,2],
            deterministic=False)

        intensifier._add_inc_run(incumbent=self.config1, run_history=self.rh)
        self.assertEqual(len(self.rh.data), 1, self.rh.data)

        intensifier._add_inc_run(incumbent=self.config1, run_history=self.rh)
        self.assertEqual(len(self.rh.data), 2, self.rh.data)
        runs = self.rh.get_runs_for_config(config=self.config1)
        # exactly one run on each instance
        self.assertIn(1, [runs[0].instance, runs[1].instance])
        self.assertIn(2, [runs[0].instance, runs[1].instance])

        intensifier._add_inc_run(incumbent=self.config1, run_history=self.rh)
        self.assertEqual(len(self.rh.data), 3, self.rh.data)

    def test_adaptive_capping(self):
        '''
            test _adapt_cutoff()
        '''
        intensifier = Intensifier(
            tae_runner=None, stats=self.stats,
            traj_logger=TrajLogger(output_dir=None, stats=self.stats),
            rng=np.random.RandomState(12345),
            instances=list(range(5)),
            deterministic=False)

        for i in range(5):
            self.rh.add(config=self.config1, cost=i+1, time=i+1,
                    status=StatusType.SUCCESS, instance_id=i,
                    seed=i,
                    additional_info=None)
        for i in range(3):
            self.rh.add(config=self.config2, cost=i+1, time=i+1,
                    status=StatusType.SUCCESS, instance_id=i,
                    seed=i,
                    additional_info=None)

        inst_seed_pairs = self.rh.get_runs_for_config(self.config1)
        # cost used by incumbent for going over all runs in inst_seed_pairs
        inc_sum_cost = sum_cost(config=self.config1, instance_seed_pairs=inst_seed_pairs,
                                    run_history=self.rh)

        cutoff = intensifier._adapt_cutoff(challenger=self.config2,
                          incumbent=self.config1,
                          run_history=self.rh,
                          inc_sum_cost=inc_sum_cost)
        # 15*1.2 - 6
        self.assertEqual(cutoff, 12)

        intensifier.cutoff = 5

        cutoff = intensifier._adapt_cutoff(challenger=self.config2,
                          incumbent=self.config1,
                          run_history=self.rh,
                          inc_sum_cost=inc_sum_cost)
        # scenario cutoff
        self.assertEqual(cutoff, 5)
