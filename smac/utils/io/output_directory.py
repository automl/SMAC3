import os
import shutil
from logging import Logger

from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def create_output_directory(
    scenario: Scenario,
    run_id: int,
    logger: Logger = None,
) -> str:
    """Create output directory for this run.

    Side effect: Adds the current output directory to the scenario object!

    Parameters
    ----------
    scenario : ~smac.scenario.scenario.Scenario
    run_id : int

    Returns
    -------
    str
    """
    if scenario.output_dir:  # type: ignore[attr-defined] # noqa F821
        output_dir = os.path.join(
            scenario.output_dir,  # type: ignore[attr-defined] # noqa F821
            "run_%d" % (run_id),
        )
    else:
        return ""
    if os.path.exists(output_dir):
        move_to = output_dir + ".OLD"
        while os.path.exists(move_to):
            move_to += ".OLD"
        shutil.move(output_dir, move_to)
        if logger is not None:
            logger.warning(
                'Output directory "%s" already exists! ' 'Moving old folder to "%s".',
                output_dir,
                move_to,
            )
    scenario.output_dir_for_this_run = output_dir
    return output_dir
