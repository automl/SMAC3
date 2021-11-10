import numpy as np
from smac.facade.smac_bb_facade import SMAC4BB
from ..test_smbo.test_epm_configuration_chooser import TestEPMChooser
from smac.runhistory.runhistory import RunHistory
from smac.tae import StatusType
from smac.optimizer.local_bo.epm_chooser_turbo import EPMChooserTurBO


class TestEPMChooserTuRBO(TestEPMChooser):
    def test_choose_next(self):
        config = self.scenario.cs.sample_configuration()
        rh = RunHistory()
        rh.add(config, 10, 10, StatusType.SUCCESS)
        smbo = SMAC4BB(scenario=self.scenario,
                       rng=np.random.RandomState(42),
                       model_type="gp",
                       smbo_kwargs={"epm_chooser": EPMChooserTurBO},
                       initial_design_kwargs={"init_budget": 0},
                       runhistory=rh
                       ).solver

        x = next(smbo.epm_chooser.choose_next()).get_array()
        self.assertEqual(x.shape, (2,))

        # remove the init configs
        smbo.epm_chooser.turbo.init_configs = []
        x = next(smbo.epm_chooser.choose_next()).get_array()

        self.assertEqual(x.shape, (2,))
