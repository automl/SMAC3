from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


"""Callbacks for SMAC.

Callbacks allow customizing the behavior of SMAC to ones needs. Currently, the list of implemented callbacks is
very limited, but they can easily be added.

How to add a new callback
=========================

1. Implement a callback class in this module. There are no restrictions on how such a callback must look like,
   but it is recommended to implement the main logic inside the `__call__` function, such as for example in
   ``IncorporateRunResultCallback``.
2. Add your callback to ``smac.smbo.optimizer.SMBO._callbacks``, using the name of your callback as the key,
   and an empty list as the value.
3. Add your callback to ``smac.smbo.optimizer.SMBO._callback_to_key``, using the callback class as the key,
   and the name as value (the name used in 2.).
4. Implement calling all registered callbacks at the correct place. This is as simple as
   ``for callback in self._callbacks['your_callback']: callback(*args, **kwargs)``, where you obviously need to
   change the callback name and signature.
"""


class IncorporateRunResultCallback:

    """Callback to react on a new run result.

    Called after the finished run is added to the runhistory.
    Optionally return `False` to (gracefully) stop the optimization.
    """

    def __call__(
            self, smbo: 'SMBO',
            run_info: RunInfo,
            result: RunValue,
            time_left: float,
    ) -> Optional[bool]:
        pass
