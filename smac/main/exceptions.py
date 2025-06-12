class ConfigurationSpaceExhaustedException(Exception):
    """Exception indicating that the configuration space is exhausted and no more configurations
    can be sampled. This is usually raised when the maximum number of configurations has been
    reached or when the configuration space has been fully explored.
    """

    pass
