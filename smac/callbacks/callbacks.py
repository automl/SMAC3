from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class IncorporateRunResultCallback:
    """Callback to react on a new run result.

    Called after the finished run is added to the runhistory. Optionally
    return `False` to (gracefully) stop the optimization.
    """

    def __call__(
        self,
        smbo: "SMBO",
        run_info: RunInfo,
        result: RunValue,
        time_left: float,
    ) -> Optional[bool]:
        """Calls the callback."""
        ...
