class ConfigurationSpaceExhaustedException(Exception):
    """Exception indicating that the configuration space is exhausted and no more configurations
    can be sampled. This is usually raised when the maximum number of configurations has been
    reached or when the configuration space has been fully explored.
    """

    pass


class AskAndTellBudgetExhaustedError(RuntimeError):
    """Raised in ask/tell mode when ``ask()`` is called after the scenario budget is exhausted
    and the respective warn mode is configured to raise.
    """

    pass
