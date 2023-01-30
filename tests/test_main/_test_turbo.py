import unittest

import numpy as np

from smac.facade.blackbox_facade import BlackBoxFacade
from smac.main.turbo import TuRBOSMBO
from smac.runhistory.runhistory import RunHistory
from smac.runner.abstract_runner import StatusType
from smac.scenario import Scenario
from smac.utils import _test_helpers


class TuRBOFacade(BlackBoxFacade):
    """A wrapper that allows to run TuRBO optimizer. Its arguments are described under smac.main.turbo.TuRBOSMBO"""

    def _init_optimizer(
        self,
        length_init=0.8,
        length_min=0.5**8,
        length_max=1.6,
        success_tol=3,
        failure_tol_min=4,
        n_init_x_params=2,
        n_candidate_max=5000,
    ) -> None:
        self.optimizer = TuRBOSMBO(
            length_init=length_init,
            length_min=length_min,
            length_max=length_max,
            success_tol=success_tol,
            failure_tol_min=failure_tol_min,
            n_init_x_params=n_init_x_params,
            n_candidate_max=n_candidate_max,
            scenario=self._scenario,
            stats=self.stats,
            runner=self.runner,
            initial_design=self.initial_design,
            runhistory=self.runhistory,
            runhistory_encoder=self.runhistory_encoder,
            intensifier=self.intensifier,
            model=self.model,
            acquisition_function=self.acquisition_function,
            acquisition_optimizer=self.acquisition_optimizer,
            random_design=self.random_design,
            seed=self.seed,
        )


def test_choose_next(make_scenario):
    cs = _test_helpers.get_branin_config_space()
    config = cs.sample_configuration()
    scenario = make_scenario(cs)
    rh = RunHistory()
    rh.add(config, 10, 10, StatusType.SUCCESS)
    tae = lambda x: x
    smbo = TuRBOFacade(scenario=scenario, target_function=tae, runhistory=rh, overwrite=True).optimizer

    x = next(smbo.ask()).get_array()
    assert x.shape == (2,)

    # remove the init configs
    smbo.turbo.init_configs = []
    x = next(smbo.ask()).get_array()

    assert x.shape == (2,)
