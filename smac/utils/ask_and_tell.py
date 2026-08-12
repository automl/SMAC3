# -*- coding: utf-8 -*-

from smac.constants import ASK_AND_TELL_VALID_WARN_MODES

def validate_warn_mode(value: str) -> str:
    """Validate that the warn_mode value is supported.

    Parameters
    ----------
    value : str
        The warn_mode value to validate.

    Returns
    -------
    str
        The validated warn_mode string.

    Raises
    ------
    ValueError
        If the provided value is not a valid warn_mode.
    """
    if value not in ASK_AND_TELL_VALID_WARN_MODES:
        raise ValueError(f"Unknown warn_mode `{value}`. Allowed: {sorted(ASK_AND_TELL_VALID_WARN_MODES)}")
    return value