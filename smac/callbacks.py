from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue


class IncorporateRunResultCallback:

    def __call__(
            self, smbo: 'SMBO',
            run_info: RunInfo,
            result: RunValue,
            time_left: float,
    ) -> None:
        pass
