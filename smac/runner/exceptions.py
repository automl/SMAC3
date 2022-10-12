class TargetAlgorithmAbortException(Exception):
    """Exception indicating that the target function suggests an ABORT of SMAC, usually because it
    assumes that all further runs will surely fail.
    """

    pass


class FirstRunCrashedException(TargetAlgorithmAbortException):
    """Exception indicating that the first run crashed (depending on options this could trigger an
    ABORT of SMAC).
    """

    pass
