import numpy as np

from smac.runhistory.encoder.encoder import (
    RunhistoryTransformer,
    RunhistoryLogScaledTransformer,
)
from smac.runhistory.encoder.boing_encoder import (
    RunHistory2EPM4CostWithRaw,
    RunHistory2EPM4ScaledLogCostWithRaw,
)
from smac.runner.abstract_runner import StatusType

from tests_old.test_runhistory.test_runhistory2epm import RunhistoryTest


class TestRH2EPMBOinG(RunhistoryTest):
    def test_cost_without_imputation(self):
        rh2epm_kwargs = dict(
            num_params=2,
            success_states=[StatusType.SUCCESS, StatusType.CRASHED, StatusType.MEMOUT],
            impute_censored_data=False,
            scenario=self.scen,
        )
        rh2epm = RunhistoryTransformer(**rh2epm_kwargs)
        rh2epm_log = RunhistoryLogScaledTransformer(**rh2epm_kwargs)

        rh2epm_with_raw = RunHistory2EPM4CostWithRaw(**rh2epm_kwargs)

        rh2epm_log_with_raw = RunHistory2EPM4ScaledLogCostWithRaw(**rh2epm_kwargs)

        self.rh.add(
            config=self.config1,
            cost=1,
            time=1,
            status=StatusType.SUCCESS,
            instance_id=23,
            seed=None,
            additional_info=None,
        )

        # rh2epm should use cost and not time field later
        self.rh.add(
            config=self.config3,
            cost=200,
            time=20,
            status=StatusType.SUCCESS,
            instance_id=1,
            seed=None,
            additional_info=None,
        )

        _, y = rh2epm.transform(self.rh)
        _, y_log = rh2epm_log.transform(self.rh)
        _, y_raw_transformed, y_raw_ = rh2epm_with_raw.transform_with_raw(self.rh)
        _, y_log_transformed, y_log_raw = rh2epm_log_with_raw.transform_with_raw(self.rh)
        # all are the raw runhistory values

        raw_values = [y_raw_transformed, y_raw_, y_log_raw]
        for raw_value in raw_values:
            np.testing.assert_array_equal(raw_value, y)

        np.testing.assert_equal(y_log, y_log_transformed)
