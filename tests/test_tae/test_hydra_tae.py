import os
import unittest
import shlex

from smac.configspace import ConfigurationSpace
from smac.tae.execute_ta_run_hydra import ExecuteTARunHydra
from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib
from smac.tae import StatusType
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TaeHydra(unittest.TestCase):

    def setUp(self):
        self.current_dir = os.getcwd()
        base_dir = os.path.split(__file__)[0]
        base_dir = os.path.join(base_dir, '..', '..')
        self.oracle = {}
        os.chdir(base_dir)

    def tearDown(self):
        os.chdir(self.current_dir)

    def test_run(self):
        '''
            running some simple algo in aclib 2.0 style
        '''
        scen = Scenario(scenario={'cs': ConfigurationSpace(),
                                  'run_obj': 'quality',
                                  'output_dir': ''}, cmd_options=None)
        stats = Stats(scen)
        self.oracle['inst'] = 0.0

        eta = ExecuteTARunHydra(
            cost_oracle=self.oracle, tae=ExecuteTARunAClib,
            ta=shlex.split("python test/test_tae/dummy_ta_wrapper_aclib.py 1"),
            stats=stats)
        status, cost, runtime, ar_info = eta.run(config={}, instance='inst', cutoff=10)
        assert status == StatusType.SUCCESS
        assert cost == 0
        assert runtime == 0

        print(status, cost, runtime)

        eta = ExecuteTARunHydra(cost_oracle=self.oracle, tae=ExecuteTARunAClib,
                                ta=shlex.split("python test/test_tae/dummy_ta_wrapper_aclib.py 2"),
                                stats=stats)
        status, cost, runtime, ar_info = eta.run(config={}, instance='inst', cutoff=10)
        assert status == StatusType.SUCCESS
        assert cost == 0, cost
        assert runtime == 0

        print(status, cost, runtime)

        eta = ExecuteTARunHydra(cost_oracle=self.oracle, tae=ExecuteTARunAClib,
                                ta=shlex.split("python test/test_tae/dummy_ta_wrapper_aclib.py 2"), stats=stats,
                                run_obj="quality")
        status, cost, runtime, ar_info = eta.run(config={}, instance='inst', cutoff=10)
        assert status == StatusType.SUCCESS
        assert cost == 0
        assert runtime == 3.0

        print(status, cost, runtime, ar_info)


if __name__ == "__main__":
    unittest.main()
