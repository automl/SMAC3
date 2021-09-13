import unittest
from unittest import mock

import numpy as np

from smac.intensification.abstract_racer import RunInfoIntent
from smac.intensification.parallel_scheduling import ParallelScheduler
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae import StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def mock_ranker(sh):
    return sh.stage, len(sh.run_tracker)


class TestParallelScheduler(unittest.TestCase):

    def test_sort_instances_by_stage(self):
        """Ensures that we prioritize the more advanced stage iteration"""

        scheduler = ParallelScheduler(
            stats=None,
            traj_logger=None,
            instances=[1, 2, 3],
            rng=np.random.RandomState(12345), deterministic=True,
        )

        scheduler._get_intensifier_ranking = mock_ranker

        def add_sh_mock(stage, config_inst_pairs):
            sh = mock.Mock()
            sh.run_tracker = []
            for i in range(config_inst_pairs):
                sh.run_tracker.append((i, i, i))
            sh.stage = stage
            return sh

        # Add more SH to make testing interesting
        instances = {}
        instances[0] = add_sh_mock(stage=1, config_inst_pairs=6)
        instances[1] = add_sh_mock(stage=1, config_inst_pairs=2)

        # We only have two configurations in the same stage.
        # In this case, we want to prioritize the one with more launched runs
        # that is zero
        self.assertEqual(
            list(scheduler._sort_instances_by_stage(instances)),
            [0, 1]
        )

        # One more instance comparison to be supper safe
        instances[2] = add_sh_mock(stage=1, config_inst_pairs=7)
        self.assertEqual(
            list(scheduler._sort_instances_by_stage(instances)),
            [2, 0, 1]
        )

        # Not let us add a more advanced stage run
        instances[3] = add_sh_mock(stage=2, config_inst_pairs=1)
        self.assertEqual(
            list(scheduler._sort_instances_by_stage(instances)),
            [3, 2, 0, 1]
        )

        # Make 1 the oldest stage
        instances[1] = add_sh_mock(stage=4, config_inst_pairs=1)
        self.assertEqual(
            list(scheduler._sort_instances_by_stage(instances)),
            [1, 3, 2, 0]
        )

        # Add a new run that's empty
        instances[4] = add_sh_mock(stage=0, config_inst_pairs=0)
        self.assertEqual(
            list(scheduler._sort_instances_by_stage(instances)),
            [1, 3, 2, 0, 4]
        )

        # Make 4 stage 4 but with not as many instances as 1
        instances[4] = add_sh_mock(stage=4, config_inst_pairs=0)
        self.assertEqual(
            list(scheduler._sort_instances_by_stage(instances)),
            [1, 4, 3, 2, 0]
        )

        # And lastly 0 -> stage 4
        instances[0] = add_sh_mock(stage=4, config_inst_pairs=0)
        self.assertEqual(
            list(scheduler._sort_instances_by_stage(instances)),
            [1, 0, 4, 3, 2]
        )

    def test_process_results(self):
        """Ensures that the results are processed by the pertinent intensifer,
        based on the source id"""
        scheduler = ParallelScheduler(
            stats=None,
            traj_logger=None,
            instances=[1, 2, 3],
            rng=np.random.RandomState(12345), deterministic=True,
        )

        scheduler.intensifier_instances = {
            0: mock.Mock(),
            1: mock.Mock(),
            2: mock.Mock(),
        }

        run_info = RunInfo(
            config=None,
            instance=0,
            instance_specific="0",
            cutoff=None,
            seed=0,
            capped=False,
            budget=0.0,
            source_id=2,
        )

        result = RunValue(
            cost=1,
            time=0.5,
            status=StatusType.SUCCESS,
            starttime=1,
            endtime=2,
            additional_info={}
        )

        scheduler.process_results(run_info=run_info, result=result, incumbent=None,
                                  run_history=None, time_bound=None)
        self.assertIsNone(scheduler.intensifier_instances[0].process_results.call_args)
        self.assertIsNone(scheduler.intensifier_instances[1].process_results.call_args)
        self.assertEqual(scheduler.intensifier_instances[2].process_results.call_args[1]['run_info'],
                         run_info)

    def test_get_next_run_wait(self):
        """Makes sure we wait if all intensifiers are busy, and no new instance got added.
        This test the case that number of workers are equal to number of instances
        """
        scheduler = ParallelScheduler(
            stats=None,
            traj_logger=None,
            instances=[1, 2, 3],
            rng=np.random.RandomState(12345), deterministic=True,
        )
        scheduler._get_intensifier_ranking = mock_ranker
        scheduler.intensifier_instances = {0: mock.Mock()}
        scheduler.intensifier_instances[0].get_next_run.return_value = (RunInfoIntent.WAIT, None)
        scheduler.intensifier_instances[0].stage = 0
        scheduler.intensifier_instances[0].run_tracker = ()

        with unittest.mock.patch(
                'smac.intensification.parallel_scheduling.ParallelScheduler._add_new_instance'
        ) as add_new_instance:
            add_new_instance.return_value = False
            intent, run_info = scheduler.get_next_run(
                challengers=None, incumbent=None, chooser=None,
                run_history=None, repeat_configs=False,
                num_workers=1
            )
            self.assertEqual(intent, RunInfoIntent.WAIT)

    def test_get_next_run_add_instance(self):
        """Makes sure we add an instance only when all other instances are waiting,
        This happens when n_workers greater than the number of instances
        """
        with unittest.mock.patch(
                'smac.intensification.parallel_scheduling.ParallelScheduler._add_new_instance'
        ) as add_new_instance:
            scheduler = ParallelScheduler(
                stats=None,
                traj_logger=None,
                instances=[1, 2, 3],
                rng=np.random.RandomState(12345), deterministic=True,
            )

            def instance_added(args):
                source_id = len(scheduler.intensifier_instances)
                scheduler.intensifier_instances[source_id] = mock.Mock()
                scheduler.intensifier_instances[source_id].get_next_run.return_value = (
                    RunInfoIntent.RUN,
                    None
                )
                return True

            add_new_instance.side_effect = instance_added
            scheduler._get_intensifier_ranking = mock_ranker
            scheduler.intensifier_instances = {0: mock.Mock()}
            scheduler.intensifier_instances[0].get_next_run.return_value = (RunInfoIntent.WAIT, None)
            scheduler.intensifier_instances[0].stage = 0
            scheduler.intensifier_instances[0].run_tracker = ()

            intent, run_info = scheduler.get_next_run(
                challengers=None, incumbent=None, chooser=None,
                run_history=None, repeat_configs=False,
                num_workers=1
            )
            self.assertEqual(len(scheduler.intensifier_instances), 2)
            self.assertEqual(intent, RunInfoIntent.RUN)


if __name__ == "__main__":
    unittest.main()
