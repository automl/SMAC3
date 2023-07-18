import smac
import numpy as np


def query_callback(solver: smac.main.smbo.SMBO, callback_type: str, key: str) -> float:
    obs = None
    for callback in solver._callbacks:
        if type(callback).__name__ == callback_type:
            if callback.history:
                obs = callback.history[-1][key]
            else:
                obs = -np.inf  # FIXME: alpha can only take 0-1 so -np.inf is not too correct
            break
    if obs is None:
        raise ValueError(f"Couldn't find the {callback_type} callback.")

    return obs
