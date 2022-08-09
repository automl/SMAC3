from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from smac.smbo import SMBO

from smac.runhistory import RunInfo, RunValue

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class IncorporateRunResultCallback:
    """Callback to react on a new run result.

    Called after the finished run is added to the runhistory. Optionally
    return `False` to (gracefully) stop the optimization.
    """

    def __call__(
        self,
        smbo: SMBO,
        run_info: RunInfo,
        result: RunValue,
        time_left: float,
    ) -> bool | None:
        """Calls the callback."""
        ...
